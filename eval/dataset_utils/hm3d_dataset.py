from eval.dataset_utils import Episode, SceneData, SemanticObject, SemanticRegion

# typing
from typing import Dict, List

# filesystem utils
import os
from os import listdir
import gzip
import json

def load_hm3d_episodes(episodes: List[Episode], scene_data: Dict[str, SceneData], object_nav_path: str):
    """
    加载HM3D episodes，支持新的R2R转换格式
    
    新增字段支持:
    - sub_instructions: 子指令列表
    - state_constraints: 状态约束
    - decisions: 决策信息
    - destination: 目标描述
    - best_seq_dists: 最优路径序列
    - radius: 目标半径
    """
    i = 0
    files = listdir(object_nav_path)
    files = sorted(files, key=str.casefold)
    
    for file in files:
        if file.endswith('.json.gz'):
            with gzip.open(os.path.join(object_nav_path, file), 'r') as f:
                json_data = json.load(f)
                
                for ep in json_data['episodes']:
                    scene_id = ep['scene_id']
                    
                    if scene_id not in scene_data:
                        scene_data_ = SceneData(scene_id, {}, {})
                        
                        if 'goals_by_category' in json_data:
                            for obj_ in json_data['goals_by_category']:
                                obj = json_data['goals_by_category'][obj_]
                                obj_name = obj[0]['object_category']
                                scene_data_.object_locations[obj_name] = []
                                scene_data_.object_ids[obj_name] = []
                                for obj_loc in obj:
                                    scene_data_.object_ids[obj_name].append(obj_loc['object_id'])
                        
                        scene_data[scene_id] = scene_data_
                    
                    episode = Episode(
                        ep['scene_id'],
                        ep.get('episode_id', i),
                        ep['start_position'],
                        ep['start_rotation'],
                        ep.get('object_goals', [ep.get('object_category', 'unknown')]),
                        ep.get('best_seq_dists', [[ep.get('info', {}).get('geodesic_distance', 0.0), [0.0, 0.0, 0.0]]])
                    )
                    
                    episode.sub_instructions = ep.get('sub_instructions', [])
                    episode.instruction_text = ep.get('instruction_text', '')
                    episode.state_constraints = ep.get('state_constraints', {})
                    episode.decisions = ep.get('decisions', {})
                    episode.destination = ep.get('destination', '')
                    episode.radius = ep.get('radius', None)
                    
                    episodes.append(episode)
                    i += 1
    
    return episodes, scene_data

def load_hm3d_objects(scene_data: Dict[str, SceneData], semantic_scene, scene_id: str):
    """
    加载HM3D场景中的物体信息和房间信息
    
    Args:
        scene_data: 场景数据字典
        semantic_scene: Habitat语义场景对象
        scene_id: 场景ID
    """
    if not scene_data[scene_id].rooms_loaded:
        for region in semantic_scene.regions:
            region_name = region.category.name()
            region_id = region.id
            
            if region_name not in scene_data[scene_id].room_locations:
                scene_data[scene_id].room_locations[region_name] = []
            
            scene_data[scene_id].room_locations[region_name].append(
                SemanticRegion(
                    region_id=region_id,
                    region_category=region_name,
                    bbox=region.aabb,
                    level_id=region.level.id if region.level else None
                )
            )
        
        scene_data[scene_id].rooms_loaded = True   #这个参数默认是False，当加载完成后设置为True
        print(f"Loaded {len(semantic_scene.regions)} regions for scene {scene_id}")
    
    for scene_obj in semantic_scene.objects:
        obj_name = scene_obj.category.name()
        for cat in scene_data[scene_id].object_locations.keys():
            if scene_obj.id in scene_data[scene_id].object_locations[cat]:
                continue
            if scene_obj.semantic_id in scene_data[scene_id].object_ids[cat]:
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
            elif obj_name in cat or cat in obj_name:
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
            elif cat == "plant" and ("flower" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
            elif cat == "sofa" and ("couch" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
    return scene_data


if __name__ == '__main__':
    eps, scene_data = load_hm3d_episodes([], {}, "converted_datasets")
    print(f"Found {len(eps)} episodes")
    scene_dist = {}
    for ep in eps:
        if ep.scene_id not in scene_dist:
            scene_dist[ep.scene_id] = 1
        else:
            scene_dist[ep.scene_id] += 1

    for sc in scene_dist:
        print(f"Scene {sc}, number of eps {scene_dist[sc]}")

    obj_counts = {}
    for ep in eps:
        for obj in ep.obj_sequence:
            if obj not in obj_counts:
                obj_counts[obj] = 1
            else:
                obj_counts[obj] += 1
    total = sum([obj_counts[obj] for obj in obj_counts])
    for obj in obj_counts:
        print(f"Object {obj}, count {obj_counts[obj]}, percentage {obj_counts[obj] / total}")
