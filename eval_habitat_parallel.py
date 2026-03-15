"""
Parallel Habitat Evaluator for single GPU environments.
Uses multiprocessing to run multiple evaluation processes concurrently,
sharing the same GPU for inference while running Habitat simulation on CPU.
"""
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import traceback

from eval.habitat_evaluator import HabitatEvaluator, Result
from eval.actor import MONActor
from config import load_eval_config
from eval.dataset_utils import HM3DDataset, HM3DMultiDataset, GibsonDataset
import bz2
import pickle
import pathlib


def worker_init(rank: int, n_workers: int):
    global worker_rank, worker_count
    worker_rank = rank
    worker_count = n_workers


def evaluate_single_episode(
    rank: int,
    episode_indices: List[int],
    config,
    scene_data_shared: Dict,
    episodes_shared: List,
    result_queue: Queue,
    progress_queue: Queue,
    n_workers: int,
    gpu_id: int = 0
) -> None:
    """
    Evaluate a subset of episodes in a separate process.
    
    Args:
        rank: Worker process rank
        episode_indices: List of episode indices to evaluate
        config: Evaluation configuration
        scene_data_shared: Shared scene data dictionary
        episodes_shared: Shared episodes list
        result_queue: Queue to put results
        progress_queue: Queue to report progress
        n_workers: Total number of workers
        gpu_id: GPU device ID to use
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        torch.cuda.set_device(0)
        
        actor = MONActor(config.EvalConf)
        
        class MinimalEvaluator:
            def __init__(self, config, actor, scene_data, episodes):
                self.config = config
                self.actor = actor
                self.scene_data = scene_data
                self.episodes = episodes
                self.max_steps = config.max_steps
                self.max_dist = config.max_dist
                self.log_rerun = config.log_rerun
                self.scene_path = config.scene_path
                self.is_gibson = config.is_gibson
                self.sim = None
                self.results_path = "results_parallel/"
                os.makedirs(self.results_path + "state", exist_ok=True)
                os.makedirs(self.results_path + "trajectories", exist_ok=True)
                os.makedirs(self.results_path + "similarities", exist_ok=True)
                
            def load_scene(self, scene_id: str):
                import habitat_sim
                from habitat_sim import ActionSpec, ActuationSpec
                
                if self.sim is not None:
                    self.sim.close()
                    
                backend_cfg = habitat_sim.SimulatorConfiguration()
                backend_cfg.scene_id = self.scene_path + scene_id
                
                if not self.is_gibson:
                    backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
                
                hfov = 90
                rgb = habitat_sim.CameraSensorSpec()
                rgb.uuid = "rgb"
                rgb.hfov = hfov
                rgb.position = np.array([0, 0.88, 0])
                rgb.sensor_type = habitat_sim.SensorType.COLOR
                res = 640
                rgb.resolution = [res, res]
                
                depth = habitat_sim.CameraSensorSpec()
                depth.uuid = "depth"
                depth.hfov = hfov
                depth.sensor_type = habitat_sim.SensorType.DEPTH
                depth.position = np.array([0, 0.88, 0])
                depth.resolution = [res, res]
                
                agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
                    move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
                    turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
                    turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
                ))
                agent_cfg.sensor_specifications = [rgb, depth]
                sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
                self.sim = habitat_sim.Simulator(sim_cfg)
                
                if not self.scene_data[scene_id].objects_loaded and not self.is_gibson:
                    self.scene_data = HM3DDataset.load_hm3d_objects(
                        self.scene_data, self.sim.semantic_scene.objects, scene_id
                    )
            
            def execute_action(self, action: Dict):
                import habitat_sim
                from habitat_sim.utils import common as utils
                
                if 'discrete' in action.keys():
                    self.sim.step(action['discrete'])
                elif 'continuous' in action.keys():
                    vel_control = habitat_sim.physics.VelocityControl()
                    vel_control.controlling_lin_vel = True
                    vel_control.lin_vel_is_local = True
                    vel_control.controlling_ang_vel = True
                    vel_control.ang_vel_is_local = True
                    
                    vel_control.angular_velocity = action['continuous']['angular']
                    vel_control.linear_velocity = action['continuous']['linear']
                    agent_state = self.sim.get_agent(0).state
                    previous_rigid_state = habitat_sim.RigidState(
                        utils.quat_to_magnum(agent_state.rotation), agent_state.position
                    )
                    
                    time_step = 1.0 / self.config.controller.control_freq
                    target_rigid_state = vel_control.integrate_transform(
                        time_step, previous_rigid_state
                    )
                    
                    end_pos = self.sim.step_filter(
                        previous_rigid_state.translation, target_rigid_state.translation
                    )
                    
                    agent_state.position = end_pos
                    agent_state.rotation = utils.quat_from_magnum(
                        target_rigid_state.rotation
                    )
                    self.sim.get_agent(0).set_state(agent_state)
                    self.sim.step_physics(time_step)
            
            def evaluate_episode(self, episode_idx: int) -> Tuple[int, Result, Dict]:
                """
                Evaluate a single episode and return results.
                Returns: (episode_id, result, stats)
                """
                from scipy.spatial.transform import Rotation as R
                from eval import get_closest_dist
                from onemap_utils import monochannel_to_inferno_rgb
                import cv2
                import rerun as rr
                
                episode = self.episodes[episode_idx]
                poses = []
                result = Result.FAILURE_OOT
                steps = 0
                
                if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                    self.load_scene(episode.scene_id)
                
                import habitat_sim
                self.sim.initialize_agent(0, habitat_sim.AgentState(
                    episode.start_position, episode.start_rotation
                ))
                self.actor.reset()
                
                current_obj = episode.obj_sequence[0]
                self.actor.set_query(current_obj)
                
                while steps < self.max_steps:
                    observations = self.sim.get_sensor_observations()
                    observations['state'] = self.sim.get_agent(0).get_state()
                    
                    pose = np.zeros((4, ))
                    pose[0] = -observations['state'].position[2]
                    pose[1] = -observations['state'].position[0]
                    pose[2] = observations['state'].position[1]
                    
                    orientation = observations['state'].rotation
                    r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                    yaw, _, _ = r.as_euler("yxz")
                    pose[3] = yaw
                    poses.append(pose)
                    
                    action, called_found = self.actor.act(observations)
                    self.execute_action(action)
                    
                    if called_found:
                        dist = get_closest_dist(
                            self.sim.get_agent(0).get_state().position[[0, 2]],
                            self.scene_data[episode.scene_id].object_locations[current_obj],
                            self.is_gibson
                        )
                        if dist < self.max_dist:
                            result = Result.SUCCESS
                        else:
                            pos = self.actor.mapper.chosen_detection
                            if pos is not None:
                                pos_metric = self.actor.mapper.one_map.px_to_metric(pos[0], pos[1])
                                dist_detect = get_closest_dist(
                                    [-pos_metric[1], -pos_metric[0]],
                                    self.scene_data[episode.scene_id].object_locations[current_obj],
                                    self.is_gibson
                                )
                                if dist_detect < self.max_dist:
                                    result = Result.FAILURE_NOT_REACHED
                                else:
                                    result = Result.FAILURE_MISDETECT
                            else:
                                result = Result.FAILURE_MISDETECT
                        break
                    
                    steps += 1
                
                poses = np.array(poses)
                
                if result == Result.FAILURE_OOT and len(poses) >= 10:
                    if np.linalg.norm(poses[-1] - poses[-10]) < 0.05:
                        result = Result.FAILURE_STUCK
                
                num_frontiers = len(self.actor.mapper.nav_goals)
                if result in [Result.FAILURE_STUCK, Result.FAILURE_OOT] and num_frontiers == 0:
                    result = Result.FAILURE_ALL_EXPLORED
                
                np.savetxt(
                    f"{self.results_path}/trajectories/poses_{episode.episode_id}.csv",
                    poses, delimiter=","
                )
                
                final_sim = (self.actor.mapper.get_map() + 1.0) / 2.0
                final_sim = final_sim[0].transpose((1, 0))
                final_sim = np.flip(final_sim, axis=0)
                final_sim = monochannel_to_inferno_rgb(final_sim)
                cv2.imwrite(
                    f"{self.results_path}/similarities/final_sim_{episode.episode_id}.png",
                    final_sim
                )
                
                with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                    f.write(str(result.value))
                
                stats = {
                    'steps': steps,
                    'object': current_obj,
                    'scene': episode.scene_id,
                    'num_frontiers': num_frontiers
                }
                
                return episode.episode_id, result, stats
        
        evaluator = MinimalEvaluator(config, actor, scene_data_shared, episodes_shared)
        
        for idx, ep_idx in enumerate(episode_indices):
            try:
                ep_id, result, stats = evaluator.evaluate_episode(ep_idx)
                result_queue.put((rank, ep_id, result, stats))
                progress_queue.put((rank, idx + 1, len(episode_indices)))
                
                print(f"[Worker {rank}] Episode {ep_id}: {result.name} | Progress: {idx+1}/{len(episode_indices)}")
            except Exception as e:
                print(f"[Worker {rank}] Error evaluating episode {ep_idx}: {e}")
                traceback.print_exc()
                result_queue.put((rank, episodes_shared[ep_idx].episode_id, Result.FAILURE_OOT, {}))
        
        if evaluator.sim is not None:
            evaluator.sim.close()
            
    except Exception as e:
        print(f"[Worker {rank}] Fatal error: {e}")
        traceback.print_exc()


class ParallelHabitatEvaluator:
    """
    Parallel evaluator that distributes episodes across multiple worker processes.
    Designed for single GPU environments where workers share the GPU.
    """
    
    def __init__(self, config, n_workers: Optional[int] = None):
        self.config = config
        self.n_workers = n_workers or max(1, mp.cpu_count() // 2)
        self.n_workers = min(self.n_workers, 4)
        
        print(f"Initializing ParallelHabitatEvaluator with {self.n_workers} workers")
        
        self.episodes, self.scene_data = self._load_episodes()
        print(f"Loaded {len(self.episodes)} episodes")
        
        self.results_path = "results_parallel/"
        os.makedirs(self.results_path + "state", exist_ok=True)
        os.makedirs(self.results_path + "trajectories", exist_ok=True)
        os.makedirs(self.results_path + "similarities", exist_ok=True)
    
    def _load_episodes(self):
        """Load episode data based on configuration."""
        episodes = []
        scene_data = {}
        
        if self.config.is_gibson:
            dataset_info_file = str(pathlib.Path(self.config.object_nav_path).parent.absolute()) + "/val_info.pbz2"
            with bz2.BZ2File(dataset_info_file, 'rb') as f:
                dataset_info = pickle.load(f)
            episodes, scene_data = GibsonDataset.load_gibson_episodes(
                episodes, scene_data, dataset_info, self.config.object_nav_path
            )
        else:
            if self.config.multi_object:
                episodes, scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(
                    episodes, scene_data, self.config.object_nav_path
                )
            else:
                episodes, scene_data = HM3DDataset.load_hm3d_episodes(
                    episodes, scene_data, self.config.object_nav_path
                )
        
        return episodes, scene_data
    
    def evaluate(self, episode_range: Optional[Tuple[int, int]] = None):
        """
        Run parallel evaluation.
        
        Args:
            episode_range: Optional tuple (start, end) to evaluate a subset of episodes
        """
        start_time = time.time()
        
        if episode_range:
            episode_indices = list(range(episode_range[0], episode_range[1]))
        else:
            episode_indices = list(range(len(self.episodes)))
        
        chunks = np.array_split(episode_indices, self.n_workers)
        chunks = [chunk.tolist() for chunk in chunks if len(chunk) > 0]
        
        manager = Manager()
        result_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        scene_data_shared = manager.dict(self.scene_data)
        episodes_shared = manager.list(self.episodes)
        
        processes = []
        for rank, chunk in enumerate(chunks):
            p = Process(
                target=evaluate_single_episode,
                args=(
                    rank,
                    chunk,
                    self.config,
                    scene_data_shared,
                    episodes_shared,
                    result_queue,
                    progress_queue,
                    self.n_workers,
                    0
                )
            )
            p.start()
            processes.append(p)
        
        results = {}
        progress = {i: 0 for i in range(len(chunks))}
        
        while any(p.is_alive() for p in processes):
            while not result_queue.empty():
                rank, ep_id, result, stats = result_queue.get()
                results[ep_id] = (result, stats)
            
            while not progress_queue.empty():
                rank, completed, total = progress_queue.get()
                progress[rank] = completed
            
            total_completed = sum(progress.values())
            total_episodes = sum(len(c) for c in chunks)
            
            elapsed = time.time() - start_time
            if total_completed > 0:
                eta = elapsed / total_completed * (total_episodes - total_completed)
                print(f"\r[Main] Progress: {total_completed}/{total_episodes} | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
            
            time.sleep(0.5)
        
        for p in processes:
            p.join()
        
        while not result_queue.empty():
            rank, ep_id, result, stats = result_queue.get()
            results[ep_id] = (result, stats)
        
        total_time = time.time() - start_time
        
        self._print_summary(results, total_time)
        
        return results
    
    def _print_summary(self, results: Dict, total_time: float):
        """Print evaluation summary."""
        success = sum(1 for r, _ in results.values() if r == Result.SUCCESS)
        total = len(results)
        
        print(f"\n{'='*60}")
        print(f"Parallel Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total episodes: {total}")
        print(f"Successful: {success}")
        print(f"Success rate: {success/total*100:.2f}%")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per episode: {total_time/total:.2f}s")
        print(f"{'='*60}")
        
        result_counts = {}
        for r, _ in results.values():
            result_counts[r.name] = result_counts.get(r.name, 0) + 1
        
        print("\nResult distribution:")
        for name, count in sorted(result_counts.items()):
            print(f"  {name}: {count} ({count/total*100:.1f}%)")


def main():
    config = load_eval_config()
    
    n_workers = getattr(config, 'n_workers', None)
    if n_workers is None:
        n_workers = min(4, max(1, mp.cpu_count() // 2))
    
    evaluator = ParallelHabitatEvaluator(config, n_workers=n_workers)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
