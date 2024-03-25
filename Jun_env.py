class CityEnviroment:
    def __init__(self):
        # 건물수 초기화
        self.init_apartments = 5
        self.init_hospitals = 3
        self.init_communication_towers = 5
        self.init_ITS = 2
        self.apartments = self.init_apartments
        self.hospitals = self.init_hospitals
        self.communication_towers = self.init_communication_towers
        self.ITS = self.init_ITS

        # 인구 자본(단위 : 만원) 만족도 주차 초기화
        self.population = 1000
        self.capital = 1000
        self.satisfaction = 20
        self.last_month_satisfaction = 20
        self.week = 0

        # 인구 연속 증가
        self.consecutive_increases = 0

        # 인구 감소율
        self.population_decay_base = 0.01

    
    def step(self, action):
        # 건물/시스템의 건설/확장 비용 , 단위 : 만원
        apartment_cost = 240
        hospital_cost = 3500
        communication_tower_cost = 2
        ITS_cost = 300

        # 행동에 따른 처리
        if action == 1 and self.capital >= apartment_cost:  # 아파트 건설
            self.apartments += 1
            self.population += int(100 * (1.03 ** max(0, self.ITS - self.init_ITS)))  # ITS 확장에 따른 인구 증가율 적용
            self.capital -= apartment_cost
        elif action == 2 and self.capital >= hospital_cost:  # 병원 건설
            self.hospitals += 1
            self.capital -= hospital_cost
        elif action == 3 and self.capital >= communication_tower_cost:  # 통신 기지국 건설
            self.communication_towers += 1
            self.capital -= communication_tower_cost
        elif action == 4 and self.capital >= ITS_cost:  # ITS 확장
            self.ITS += 1
            self.capital -= ITS_cost
