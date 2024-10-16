# Game Control Using RL methods 

本專案是使用 Q-learning 演算法訓練的 Flappy Bird 遊戲智能體。透過強化學習，智能體能夠在多次遊戲迭代中學習如何躲避管道，並儘可能長時間保持飛行。

## Flappy Bird 環境簡介

Flappy Bird 是一款橫向卷軸遊戲，玩家需點擊螢幕讓小鳥向上飛行，避免撞到管道，並成功穿過管道之間的縫隙。

其中玩家可感知到的狀態：

	•	player_y                      玩家y軸的座標
	•	player_vel                    玩家速度
	•	next_pipe_dist_to_player      下個管道與玩家的距離
	•	next_pipe_top_y               下個管道的上邊界
	•	next_pipe_bottom_y            下個管道的下邊界
	•	next_next_pipe_dist_to_player 下下個管道與玩家的距離
	•	next_next_pipe_top_y          下下個管道的上邊界
	•	next_next_pipe_bottom_y       下下個管道的下邊界

其中玩家可進行的動作：

	•	動作0  向上加速              
	•	動作1  不動

## Q-learning 算法簡介

Q-learning 是一種基於表格的強化學習演算法，智能體透過探索環境、執行動作、獲取獎勵，並更新策略，逐步學習完成任務的最佳方式。其核心公式為：
```math
Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
```
其中：

	•	s 為當前狀態
	•	a 為當前動作
	•	r 為執行動作後的獎勵
	•	s' 為下一狀態
	•	a' 為下一動作
	•	α 為學習率
	•	γ 為折扣因子

透過不斷更新 $Q$ 表，智能體會逐漸收斂至最優策略。