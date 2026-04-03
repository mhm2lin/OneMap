# eval utils
from eval import get_closest_dist, FMMPlanner
from eval.actor import Actor
from eval.dataset_utils.gibson_dataset import load_gibson_episodes
from mapping import rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *

# os / filsystem
import bz2
import os
from os import listdir
import gzip
import json
import pathlib

# cv2
import cv2

# numpy
import numpy as np

# skimage
import skimage


# dataclasses
from dataclasses import dataclass

# quaternion
import quaternion

# typing
from typing import Tuple, List, Dict, Optional
import enum

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# tabulate
from tabulate import tabulate

# rerun
import rerun as rr

# pandas
import pandas as pd

# pickle
import pickle

# scipy
from scipy.spatial.transform import Rotation as R


class Result(enum.Enum):
    SUCCESS = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6


class SubTaskManager:
    """
    子任务管理器
    负责管理导航任务中的子任务序列，从decisions中提取目标物体
    """
    
    def __init__(self, episode):
        """
        初始化子任务管理器
        
        Args:
            episode: Episode对象，包含sub_instructions, decisions, state_constraints等
        """
        self.episode = episode
        self.sub_instructions = episode.sub_instructions
        self.decisions = episode.decisions
        self.state_constraints = episode.state_constraints
        self.current_subtask_idx = 0
        self.total_subtasks = len(self.sub_instructions)
        
    def get_current_subtask(self) -> Optional[Dict]:
        """
        获取当前子任务信息
        
        Returns:
            包含sub_instruction, target_objects, directions等信息的字典
            如果没有更多子任务，返回None
        """
        if self.current_subtask_idx >= self.total_subtasks:
            return None
        
        sub_idx = str(self.current_subtask_idx)
        sub_instruction = self.sub_instructions[self.current_subtask_idx] if self.current_subtask_idx < len(self.sub_instructions) else ""
        
        decision = self.decisions.get(sub_idx, {})
        constraints = self.state_constraints.get(sub_idx, [])
        
        target_objects = self._extract_target_objects(decision, constraints)
        directions = decision.get('directions', [])
        
        subtask = {
            'subtask_id': self.current_subtask_idx,
            'sub_instruction': sub_instruction,
            'target_objects': target_objects,
            'directions': directions,
            'decision': decision,
            'constraints': constraints
        }
        
        return subtask
    
    def _extract_target_objects(self, decision: Dict, constraints: List) -> List[str]:
        """
        从decision和constraints中提取目标物体或房间
        
        Args:
            decision: decisions字典中对应子任务的决策信息
            constraints: state_constraints中对应子任务的约束列表
            
        Returns:
            目标物体或房间名称列表
        """
        ROOM_NAMES = {
            'bedroom', 'bathroom', 'kitchen', 'living room', 'living_room',
            'dining room', 'dining_room', 'office', 'study', 'garage',
            'hallway', 'hall', 'lobby', 'entrance', 'closet', 'pantry',
            'laundry room', 'laundry_room', 'basement', 'attic', 'garage',
            'porch', 'balcony', 'terrace', 'garden', 'yard', 'room', 'stairs',
            'staircase', 'stairway', 'corridor', 'passage', 'passageway'
        }
        
        target_objects = []
        
        landmarks = decision.get('landmarks', [])
        for landmark in landmarks:
            if isinstance(landmark, list) and len(landmark) > 0:
                obj_name = landmark[0]
                if obj_name and obj_name not in target_objects:
                    target_objects.append(obj_name)
        
        for constraint in constraints:
            if isinstance(constraint, list) and len(constraint) >= 2:
                constraint_type = constraint[0]
                constraint_value = constraint[1]
                
                if constraint_type == 'object constraint':
                    if constraint_value not in target_objects:
                        target_objects.append(constraint_value)
        
        return target_objects
    
    def has_valid_target(self) -> bool:
        """
        检查当前子任务是否有有效的目标物体
        
        Returns:
            True如果有目标物体，False否则
        """
        subtask = self.get_current_subtask()
        if subtask is None:
            return False
        
        return len(subtask['target_objects']) > 0
    
    def advance_to_next_valid_subtask(self) -> Optional[Dict]:
        """
        前进到下一个有效的子任务（有目标物体的子任务）
        跳过没有目标物体的子任务
        
        Returns:
            下一个有效的子任务信息，如果没有则返回None
        """
        while self.current_subtask_idx < self.total_subtasks:
            self.current_subtask_idx += 1
            
            if self.has_valid_target():
                return self.get_current_subtask()
        
        return None
    
    def advance_subtask(self) -> bool:
        """
        前进到下一个子任务
        
        Returns:
            True如果还有下一个子任务，False如果没有
        """
        self.current_subtask_idx += 1
        return self.current_subtask_idx < self.total_subtasks
    
    def is_finished(self) -> bool:
        """
        检查是否已完成所有子任务
        
        Returns:
            True如果已完成，False否则
        """
        return self.current_subtask_idx >= self.total_subtasks
    
    def get_progress(self) -> Tuple[int, int]:
        """
        获取当前进度
        
        Returns:
            (当前子任务索引, 总子任务数)
        """
        return (self.current_subtask_idx, self.total_subtasks)


class HabitatEvaluator:
    def __init__(self,
                 config: EvalConf,
                 actor: Actor,
                 ) -> None:
        self.config = config
        self.multi_object = config.multi_object
        self.max_steps = config.max_steps
        self.max_dist = config.max_dist
        self.controller = config.controller
        self.mapping = config.mapping
        self.planner = config.planner
        self.log_rerun = config.log_rerun
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
        self.is_gibson = config.is_gibson
        self.use_subtask_manager = config.use_subtask_manager if hasattr(config, 'use_subtask_manager') else False

        self.sim = None
        self.actor = actor
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.control_frequency = config.controller.control_freq
        self.max_vel = config.controller.max_vel
        self.max_ang_vel = config.controller.max_ang_vel
        self.time_step = 1.0 / self.control_frequency
        if self.is_gibson:
            dataset_info_file = str(pathlib.Path(self.object_nav_path).parent.absolute()) + \
                                "/val_info.pbz2".format(split="val")
            with bz2.BZ2File(dataset_info_file, 'rb') as f:
                self.dataset_info = pickle.load(f)
        else:
            self.dataset_info = None
        if self.is_gibson:
            self.episodes, self.scene_data = GibsonDataset.load_gibson_episodes(self.episodes,
                                                                                self.scene_data,
                                                                                self.dataset_info,
                                                                                self.object_nav_path)
        else:
            if self.multi_object:
                self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                           self.scene_data,
                                                                                           self.object_nav_path)
            else:
                self.episodes, self.scene_data = HM3DDataset.load_hm3d_episodes(self.episodes,
                                                                                self.scene_data,
                                                                                self.object_nav_path)
        if self.actor is not None:
            self.logger = rerun_logger.RerunLogger(self.actor.mapper, self.log_rerun, "results/output.rrd", debug=False) if self.log_rerun else None      
        self.results_path = "/home/finn/active/MON/results_gibson" if self.is_gibson else "results/"

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path + scene_id
        if self.is_gibson:
            pass
        else:
            backend_cfg.scene_dataset_config_file = self.scene_path + "mp3d.scene_dataset_config.json"

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
        if self.scene_data[scene_id].objects_loaded:
            return
        if not self.is_gibson:
            self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene, scene_id)
        else:
            self.scene_data = GibsonDataset.load_gibson_objects(self.scene_data, self.dataset_info, scene_id)


    def execute_action(self, action: Dict):
        if 'discrete' in action.keys():
            self.sim.step(action['discrete'])

        elif 'continuous' in action.keys():
            self.vel_control.angular_velocity = action['continuous']['angular']
            self.vel_control.linear_velocity = action['continuous']['linear']
            agent_state = self.sim.get_agent(0).state
            previous_rigid_state = habitat_sim.RigidState(
                utils.quat_to_magnum(agent_state.rotation), agent_state.position
            )

            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            end_pos = self.sim.step_filter(
                previous_rigid_state.translation, target_rigid_state.translation
            )

            agent_state.position = end_pos
            agent_state.rotation = utils.quat_from_magnum(
                target_rigid_state.rotation
            )
            self.sim.get_agent(0).set_state(agent_state)
            self.sim.step_physics(self.time_step)

    def read_results(self, path, sort_by):
        state_dir = os.path.join(path, 'state')
        state_results = {}
        object_query = {}
        scene_name = {}
        spl = {}

        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        for filename in os.listdir(state_dir):
            if filename.startswith('state_') and filename.endswith('.txt'):
                try:
                    experiment_num = int(filename[6:-4])
                    with open(os.path.join(state_dir, filename), 'r') as file:
                        content = file.read().strip()

                    state_value = int(content)
                    state_results[experiment_num] = state_value
                    object_query[experiment_num] = self.episodes[experiment_num].obj_sequence[0]
                    scene_name[experiment_num] = self.episodes[experiment_num].scene_id
                    poses = np.genfromtxt(os.path.join(pose_dir, "poses_" + str(experiment_num) + ".csv"), delimiter=",")
                    deltas = poses[1:, :2] - poses[:-1, :2]
                    distance_traveled = np.linalg.norm(deltas, axis=1).sum()
                    if state_value == 1:
                        spl[experiment_num] = self.episodes[experiment_num].best_dist / max(self.episodes[experiment_num].best_dist, distance_traveled)
                    else:
                        spl[experiment_num] = 0
                    if self.episodes[experiment_num].episode_id != experiment_num:
                        print(f"Warning, exerpiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                except ValueError:
                    print(f"Warning: Skipping {filename} due to invalid format")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        dict_res = {"state": state_results, "obj" : object_query, "scene" : scene_name, "spl" : spl}
        data = pd.DataFrame.from_dict(dict_res)

        states = data["state"].unique()

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})
            avg_spl = group['spl'].mean()
            result['Average SPL'] = avg_spl
            return result

        object_results = data.groupby('obj').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'obj': 'Object'})

        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        return data

    def evaluate(self):
        success = 0
        n_eps = 0
        success_per_obj = {}
        obj_count = {}
        results = []
        
        for n_ep, episode in enumerate(self.episodes):
            poses = []
            results.append(Result.FAILURE_OOT)
            steps = 0
            if n_ep in self.exclude_ids:
                continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)
            
            self.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            self.actor.reset()
            
            if self.use_subtask_manager and hasattr(episode, 'decisions') and episode.decisions:
                subtask_manager = SubTaskManager(episode)
                subtask = subtask_manager.get_current_subtask()
                
                if subtask is None or not subtask['target_objects']:
                    print(f"Episode {episode.episode_id}: No valid subtasks with target objects, skipping...")
                    results[n_ep] = Result.FAILURE_OOT
                    continue
                
                current_obj = subtask['target_objects'][0]
                
                is_room_target = current_obj.lower() in {
                    'bedroom', 'bathroom', 'kitchen', 'living room', 'living_room',
                    'dining room', 'dining_room', 'office', 'study', 'garage',
                    'hallway', 'hall', 'lobby', 'entrance', 'closet', 'pantry',
                    'laundry room', 'laundry_room', 'basement', 'attic',
                    'porch', 'balcony', 'terrace', 'garden', 'yard', 'room', 'stairs',
                    'staircase', 'stairway', 'corridor', 'passage', 'passageway'
                }
                
                target_locations = (current_obj in self.scene_data[episode.scene_id].object_locations or 
                                   current_obj in self.scene_data[episode.scene_id].room_locations)
                
                if not target_locations:
                    print(f"Episode {episode.episode_id}: Target '{current_obj}' not found in scene data, skipping...")
                    results[n_ep] = Result.FAILURE_OOT
                    continue
                
                print(f"Episode {episode.episode_id}: Starting with subtask 0, target: {current_obj} ({'room' if is_room_target else 'object'})")
                print(f"  Sub-instruction: {subtask['sub_instruction']}")
            else:
                current_obj_id = 0
                current_obj = episode.obj_sequence[current_obj_id]
                
                if current_obj not in self.scene_data[episode.scene_id].object_locations:
                    print(f"Episode {episode.episode_id}: Target object '{current_obj}' not found in scene data, skipping...")
                    results[n_ep] = Result.FAILURE_OOT
                    continue
            
            if current_obj not in success_per_obj:
                success_per_obj[current_obj] = 0
                obj_count[current_obj] = 1
            else:
                obj_count[current_obj] += 1
            
            self.actor.set_query(current_obj)
            
            target_locations = []
            if current_obj in self.scene_data[episode.scene_id].object_locations:
                target_locations = self.scene_data[episode.scene_id].object_locations[current_obj]
            elif current_obj in self.scene_data[episode.scene_id].room_locations:
                target_locations = self.scene_data[episode.scene_id].room_locations[current_obj]
            
            if self.log_rerun:
                pts = []
                if target_locations:
                    for obj in target_locations:
                        if not self.is_gibson:
                            pt = obj.bbox.center[[0, 2]]
                            pt = (-pt[1], -pt[0])
                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                        else:
                            for pt_ in obj:
                                pt = (pt_[0], pt_[1])
                                pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                    pts = np.array(pts)
                    rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))

            while steps < self.max_steps:
                if self.use_subtask_manager and hasattr(episode, 'decisions') and episode.decisions:
                    if subtask_manager.is_finished():
                        print(f"Episode {episode.episode_id}: All subtasks completed!")
                        results[n_ep] = Result.SUCCESS
                        success += 1
                        break
                
                observations = self.sim.get_sensor_observations()
                observations['state'] = self.sim.get_agent(0).get_state()
                pose = np.zeros((4, ))
                pose[0] = -observations['state'].position[2]
                pose[1] = -observations['state'].position[0]
                pose[2] = observations['state'].position[1]
                
                orientation = observations['state'].rotation
                q0 = orientation.x
                q1 = orientation.y
                q2 = orientation.z
                q3 = orientation.w
                r = R.from_quat([q0, q1, q2, q3])
                yaw, _, _1 = r.as_euler("yxz")
                pose[3] = yaw

                poses.append(pose)
                if self.log_rerun:
                    cam_x = -self.sim.get_agent(0).get_state().position[2]
                    cam_y = -self.sim.get_agent(0).get_state().position[0]
                    rr.log("camera/rgb", rr.Image(observations["rgb"]).compress(jpeg_quality=50))
                    self.logger.log_pos(cam_x, cam_y)
                
                action, called_found = self.actor.act(observations)
                self.execute_action(action)
                
                if self.log_rerun:
                    self.logger.log_map()

                if called_found:
                    target_locations = []
                    if current_obj in self.scene_data[episode.scene_id].object_locations:
                        target_locations = self.scene_data[episode.scene_id].object_locations[current_obj]
                    elif current_obj in self.scene_data[episode.scene_id].room_locations:
                        target_locations = self.scene_data[episode.scene_id].room_locations[current_obj]
                    
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            target_locations, self.is_gibson)
                    
                    if dist < self.max_dist:
                        print(f"Episode {episode.episode_id}: Target '{current_obj}' found!")
                        
                        if self.use_subtask_manager and hasattr(episode, 'decisions') and episode.decisions:
                            subtask_manager.advance_subtask()
                            
                            while not subtask_manager.is_finished():
                                next_subtask = subtask_manager.get_current_subtask()
                                
                                if next_subtask and next_subtask['target_objects']:
                                    current_obj = next_subtask['target_objects'][0]
                                    
                                    target_locations = []
                                    if current_obj in self.scene_data[episode.scene_id].object_locations:
                                        target_locations = self.scene_data[episode.scene_id].object_locations[current_obj]
                                    elif current_obj in self.scene_data[episode.scene_id].room_locations:
                                        target_locations = self.scene_data[episode.scene_id].room_locations[current_obj]
                                    
                                    if not target_locations:
                                        print(f"  Target '{current_obj}' not found in scene data, skipping subtask...")
                                        subtask_manager.advance_subtask()
                                        continue
                                    
                                    current_idx, total = subtask_manager.get_progress()
                                    print(f"  Advancing to subtask {current_idx}/{total}, target: {current_obj}")
                                    print(f"  Sub-instruction: {next_subtask['sub_instruction']}")
                                    
                                    if current_obj not in success_per_obj:
                                        success_per_obj[current_obj] = 0
                                        obj_count[current_obj] = 1
                                    else:
                                        obj_count[current_obj] += 1
                                    
                                    self.actor.set_query(current_obj)
                                    break
                                else:
                                    print(f"  Subtask {subtask_manager.current_subtask_idx} has no valid target, skipping...")
                                    subtask_manager.advance_subtask()
                            else:
                                print(f"Episode {episode.episode_id}: All subtasks completed successfully!")
                                results[n_ep] = Result.SUCCESS
                                success += 1
                                break
                        else:
                            results[n_ep] = Result.SUCCESS
                            success += 1
                            success_per_obj[current_obj] += 1
                            break
                    else:
                        pos = self.actor.mapper.chosen_detection
                        pos_metric = self.actor.mapper.one_map.px_to_metric(pos[0], pos[1])
                        target_locations = []
                        if current_obj in self.scene_data[episode.scene_id].object_locations:
                            target_locations = self.scene_data[episode.scene_id].object_locations[current_obj]
                        elif current_obj in self.scene_data[episode.scene_id].room_locations:
                            target_locations = self.scene_data[episode.scene_id].room_locations[current_obj]
                        dist_detect = get_closest_dist([-pos_metric[1], -pos_metric[0]],
                                            target_locations, self.is_gibson)
                        if dist_detect < self.max_dist:
                            results[n_ep] = Result.FAILURE_NOT_REACHED
                        else:
                            results[n_ep] = Result.FAILURE_MISDETECT
                        print(f"Target not found! Dist {dist}, detect dist: {dist_detect}.")
                        break

                if steps % 100 == 0:
                    target_locations = []
                    if current_obj in self.scene_data[episode.scene_id].object_locations:
                        target_locations = self.scene_data[episode.scene_id].object_locations[current_obj]
                    elif current_obj in self.scene_data[episode.scene_id].room_locations:
                        target_locations = self.scene_data[episode.scene_id].room_locations[current_obj]
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            target_locations, self.is_gibson)
                    if self.use_subtask_manager and hasattr(episode, 'decisions') and episode.decisions:
                        current_idx, total = subtask_manager.get_progress()
                        print(f"Step {steps}, Subtask {current_idx}/{total}, Target: {current_obj}, Episode: {episode.episode_id}, Dist: {dist:.2f}m")
                    else:
                        print(f"Step {steps}, current target: {current_obj}, episode_id: {episode.episode_id}, distance to closest target: {dist}")
                steps += 1
            
            poses = np.array(poses)
            if results[n_ep] == Result.FAILURE_OOT and len(poses) >= 10 and np.linalg.norm(poses[-1] - poses[-10]) < 0.05:
                results[n_ep] = Result.FAILURE_STUCK

            num_frontiers = len(self.actor.mapper.nav_goals)
            np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}.csv", poses, delimiter=",")
            
            final_sim = (self.actor.mapper.get_map() + 1.0) / 2.0
            final_sim = final_sim[0]
            final_sim = final_sim.transpose((1, 0))
            final_sim = np.flip(final_sim, axis=0)
            final_sim = monochannel_to_inferno_rgb(final_sim)
            cv2.imwrite(f"{self.results_path}/similarities/final_sim_{episode.episode_id}.png", final_sim)
            
            if (results[n_ep] == Result.FAILURE_STUCK or results[n_ep] == Result.FAILURE_OOT) and num_frontiers == 0:
                results[n_ep] = Result.FAILURE_ALL_EXPLORED
            
            print(f"Overall success: {success / (n_eps)}, per object: ")
            for obj in success_per_obj.keys():
                print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            print(
                f"Result distribution: successes: {results.count(Result.SUCCESS)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}")
            
            with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                f.write(str(results[n_ep].value))
