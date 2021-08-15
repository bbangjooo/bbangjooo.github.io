---
layout: post
author: bbangjo
title: Wide&Deep, DeepFM 리뷰
date: 2021-08-15
thumbnail: https://user-images.githubusercontent.com/51329156/100213332-d35ee480-2f51-11eb-9d0b-20b17ed189d8.png
category: ['Recommendation System']
comments: true
usemathjax: true
summary: Recommendation System Intro
permalink: /blog/wddfm

---

# Wide&Deep, DeepFM 리뷰

2016년 구글이 발표한 [Wide&Deep](https://arxiv.org/abs/1606.07792), 그리고 2017년에 발표된 [DeepFM](https://arxiv.org/abs/1703.04247) 두 모델이 다루고자 하는 문제도 비슷하고 아이디어도 겹치는 점이 있어 같이 다뤄보겠습니다. 'Why(What problem)?', 'How?' 질문에 대한 답을 찾아내는 데 집중하고 구체적인 implementation에 관해서는 패스하도록 하겠습니다. 

## Wide&Deep

이 논문은 너무 쉽게 씌여져서 부담없이 읽으실 수 있을 것 같아요. 이 논문에서 풀고자(개선하고자) 하는 문제는 '구글 플레이의 추천 앱 랭킹 성능 개선'입니다. 즉 CTR을 높이는 것이 목표입니다.

### RECOMMENDER SYSTEM OVERVIEW

![image](https://user-images.githubusercontent.com/51329156/129478615-c56a6863-e231-445e-80bf-2e7bf54ea8d0.png)

위 그림은 추천 시스템의 overview 입니다. **Retrieval** 과정에서 user의 검색 쿼리를 바탕으로 DB에서 후보 앱들을 추출해냅니다. 그리고 **Ranking** 과정에서 앱들의 점수를 매기고, 그 점수를 바탕으로 랭킹을 매기게 됩니다. 여기서 말하는 점수는 $$P(y|x)$$ 즉 user 정보, context 정보 등을 포함한  x(국적, 언어, ..)를 바탕으로 y앱에 action할 확률입니다.

