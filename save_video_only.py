import os
import torch
import gymnasium as gym
from Lunarlander_RF import PolicyNetwork, record_video

def main():
    # 환경 초기화
    env = gym.make("LunarLander-v3", continuous=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 네트워크 생성
    policy = PolicyNetwork(state_dim, action_dim)
    
    # 저장된 가중치 불러오기
    if os.path.exists("best_policy.pth"):
        policy.load_state_dict(torch.load("best_policy.pth"))
        print("✅ [Success] 'best_policy.pth' 모델을 성공적으로 불러왔습니다.")
        
        os.makedirs("videos", exist_ok=True)
        
        # 비디오 녹화 함수 실행 (Lunarlander_RF.py 내부 함수 활용)
        print("🎥 비디오 녹화를 시작합니다...")
        record_video(policy, state_dim)
        print("🎬 완료! 'videos' 폴더를 확인해 보세요.")
    else:
        print("❌ [Error] 'best_policy.pth' 파일이 존재하지 않습니다.")

if __name__ == "__main__":
    main()
