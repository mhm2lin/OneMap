#!/usr/bin/env python3
"""
R2R数据集转换为OneMap多物体导航数据集格式

保留字段:
- start_position: 从R2R数据集获取
- start_rotation: 从R2R数据集获取
- scene_id: 从R2R数据集获取
- episode_id: 从R2R数据集获取
- sub_instructions: 从LLM回复获取
- instruction_text: 从R2R数据集获取
- best_seq_dists: 从R2R数据集的geodesic_distance和goals.position获取
- radius: 从R2R数据集的goals.radius获取
- state_constraints: 从LLM回复获取
- decisions: 从LLM回复获取
- destination: 从LLM回复获取
"""

import gzip
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class OneMapEpisode:
    start_position: List[float]
    start_rotation: List[float]
    scene_id: str
    episode_id: int
    sub_instructions: List[str] = field(default_factory=list)
    instruction_text: str = ""
    best_seq_dists: List[Tuple[float, List[float]]] = field(default_factory=list)
    radius: Optional[float] = None
    state_constraints: Dict[str, List] = field(default_factory=dict)
    decisions: Dict[str, Dict] = field(default_factory=dict)
    destination: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "start_position": self.start_position,
            "start_rotation": self.start_rotation,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "sub_instructions": self.sub_instructions,
            "instruction_text": self.instruction_text,
            "best_seq_dists": self.best_seq_dists,
            "radius": self.radius,
            "state_constraints": self.state_constraints,
            "decisions": self.decisions,
            "destination": self.destination
        }


class R2RToOneMapConverter:
    
    def __init__(self, 
                 r2r_data_path: str,
                 llm_reply_path: str,
                 output_path: str):
        self.r2r_data_path = r2r_data_path
        self.llm_reply_path = llm_reply_path
        self.output_path = output_path
        
        self.r2r_data = None
        self.llm_replies = None
        self.conversion_stats = {
            'total_episodes': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'missing_goals': 0,
            'missing_geodesic': 0,
            'missing_state_constraints': 0,
            'missing_decisions': 0
        }
    
    def load_data(self):
        print("Loading R2R data...")
        with gzip.open(self.r2r_data_path, 'r') as f:
            self.r2r_data = json.load(f)
        
        print("Loading LLM replies...")
        with open(self.llm_reply_path, 'r') as f:
            self.llm_replies = json.load(f)
        
        print(f"Loaded {len(self.r2r_data.get('episodes', []))} R2R episodes")
        print(f"Loaded {len(self.llm_replies)} LLM replies")
    
    def extract_best_seq_dists_and_radius(self, r2r_episode: Dict) -> Tuple[List[Tuple[float, List[float]]], Optional[float]]:
        """
        从R2R episode中提取best_seq_dists和radius
        
        Returns:
            best_seq_dists: [(geodesic_distance, goal_position)]
            radius: 目标半径
        """
        best_seq_dists = []
        radius = None
        
        goals = r2r_episode.get('goals', [])
        
        if not goals:
            print(f"Warning: No goals found for episode {r2r_episode.get('episode_id', -1)}")
            self.conversion_stats['missing_goals'] += 1
            return best_seq_dists, radius
        
        geodesic_distance = None
        
        if 'info' in r2r_episode and 'geodesic_distance' in r2r_episode['info']:
            geodesic_distance = r2r_episode['info']['geodesic_distance']
        elif 'geodesic_distance' in r2r_episode:
            geodesic_distance = r2r_episode['geodesic_distance']
        
        for goal in goals:
            position = goal.get('position', None)
            goal_radius = goal.get('radius', None)
            
            if position is not None:
                if geodesic_distance is not None:
                    best_seq_dists.append([float(geodesic_distance), position])
                else:
                    best_seq_dists.append([0.0, position])
                    if len(best_seq_dists) == 1:
                        self.conversion_stats['missing_geodesic'] += 1
            
            if goal_radius is not None and radius is None:
                radius = goal_radius
        
        return best_seq_dists, radius
    
    def convert_episode(self, r2r_episode: Dict) -> Optional[OneMapEpisode]:
        episode_id = r2r_episode.get('episode_id', -1)
        
        llm_reply = self.llm_replies.get(str(episode_id), {})
        
        if not llm_reply:
            print(f"Warning: No LLM reply found for episode {episode_id}")
            return None
        
        sub_instructions = llm_reply.get('sub-instructions', [])
        
        state_constraints = llm_reply.get('state-constraints', {})
        if not state_constraints:
            self.conversion_stats['missing_state_constraints'] += 1
        
        decisions = llm_reply.get('decisions', {})
        if not decisions:
            self.conversion_stats['missing_decisions'] += 1
        
        destination = llm_reply.get('destination', '')
        
        start_position = r2r_episode.get('start_position', [0.0, 0.0, 0.0])
        start_rotation = r2r_episode.get('start_rotation', [0.0, 0.0, 0.0, 1.0])
        
        if len(start_position) == 2:
            start_position = [start_position[0], 0.0, start_position[1]]
        
        scene_id = r2r_episode.get('scene_id', '')
        
        instruction_text = r2r_episode.get('instruction', {}).get('instruction_text', '')
        
        best_seq_dists, radius = self.extract_best_seq_dists_and_radius(r2r_episode)
        
        onemap_episode = OneMapEpisode(
            start_position=start_position,
            start_rotation=start_rotation,
            scene_id=scene_id,
            episode_id=episode_id,
            sub_instructions=sub_instructions,
            instruction_text=instruction_text,
            best_seq_dists=best_seq_dists,
            radius=radius,
            state_constraints=state_constraints,
            decisions=decisions,
            destination=destination
        )
        
        return onemap_episode
    
    def convert_all(self, max_episodes: Optional[int] = None) -> List[Dict]:
        episodes = self.r2r_data.get('episodes', [])
        
        if max_episodes:
            episodes = episodes[:max_episodes]
        
        onemap_episodes = []
        self.conversion_stats['total_episodes'] = len(episodes)
        
        for i, r2r_episode in enumerate(episodes):
            episode_id = r2r_episode.get('episode_id', -1)
            
            if (i + 1) % 100 == 0:
                print(f"Processing episode {i+1}/{len(episodes)} (ID: {episode_id})")
            
            try:
                onemap_episode = self.convert_episode(r2r_episode)
                
                if onemap_episode:
                    onemap_episodes.append(onemap_episode.to_dict())
                    self.conversion_stats['successful_conversions'] += 1
                else:
                    self.conversion_stats['failed_conversions'] += 1
                    
            except Exception as e:
                print(f"Error converting episode {episode_id}: {e}")
                self.conversion_stats['failed_conversions'] += 1
        
        return onemap_episodes
    
    def save(self, onemap_episodes: List[Dict]):
        output_data = {
            "episodes": onemap_episodes,
            "metadata": {
                "total_episodes": len(onemap_episodes),
                "source": "R2R_VLNCE",
                "converter": "R2RToOneMapConverter",
                "conversion_stats": self.conversion_stats
            }
        }
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        if self.output_path.endswith('.gz'):
            with gzip.open(self.output_path, 'wt') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            with open(self.output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nConversion completed!")
        print(f"Output saved to: {self.output_path}")
        print(f"\nStatistics:")
        print(f"  Total R2R episodes: {self.conversion_stats['total_episodes']}")
        print(f"  Successful conversions: {self.conversion_stats['successful_conversions']}")
        print(f"  Failed conversions: {self.conversion_stats['failed_conversions']}")
        print(f"  Missing goals: {self.conversion_stats['missing_goals']}")
        print(f"  Missing geodesic distance: {self.conversion_stats['missing_geodesic']}")
        print(f"  Missing state_constraints: {self.conversion_stats['missing_state_constraints']}")
        print(f"  Missing decisions: {self.conversion_stats['missing_decisions']}")
    
    def run(self, max_episodes: Optional[int] = None):
        self.load_data()
        onemap_episodes = self.convert_all(max_episodes)
        self.save(onemap_episodes)
        return onemap_episodes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert R2R dataset to OneMap format')
    parser.add_argument('--r2r_data', type=str, 
                        default='/root/autodl-tmp/CA-Nav-code/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz',
                        help='Path to R2R dataset')
    parser.add_argument('--llm_reply', type=str,
                        default='/root/autodl-tmp/CA-Nav-code/LLM_REPLYS_VAL_UNSEEN/llm_reply_valunseen1839.json',
                        help='Path to LLM replies from CA-Nav')
    parser.add_argument('--output', type=str,
                        default='/root/autodl-tmp/converted_datasets/r2r_to_onemap_val_unseen.json.gz',
                        help='Output path for OneMap format dataset')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes to convert (for testing)')
    
    args = parser.parse_args()
    
    converter = R2RToOneMapConverter(
        r2r_data_path=args.r2r_data,
        llm_reply_path=args.llm_reply,
        output_path=args.output
    )
    
    converter.run(max_episodes=args.max_episodes)
