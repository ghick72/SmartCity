class SmartCityEnvironment():
    def __init__(self):
        # 초기 상태
        self.capital = 100000  # 초기 자본
        self.maintenance = 0 # 유지 비용
        self.income = 0 # 수익
        self.capacity_population = 0  # 수용 가능 인원
        self.population = 0  # 인구수
        self.attrition_rate = 0.001  # 초기 이탈율
        self.influx_rate = 0.05  # 초기 유입률 
        self.num_apartment = 0  # 주거공간 개수
        self.num_base_station = 0  # 기지국 개수
        self.num_ITS = 0  # ITS 개수
        self.num_hospitals = 0  # 병원 개수
        self.happiness = 60  # 행복도
        self.last_reward_population_level = 0
        self.step_count = 0
        
    def step(self, action):
        if self.step_count == 1040: # 1에피소드 완료
            return self.get_state(), 0, False , {}
        self.step_count += 1

        # if self.population > 50000:
        #     return self.get_state(), 100, False, {}
            
        
        self.apartment_cost = 6000
        self.base_station_cost = 5000
        self.ITS_cost = 2500
        self.hospitals_cost = 6500
        
        
        if action == 0 and self.capital > self.apartment_cost:
            self.capital -= self.apartment_cost
            self.num_apartment += 1
            self.capacity_population += 100
        if action == 1 and self.capital > self.base_station_cost:
            self.capital -= self.base_station_cost
            self.num_base_station += 1
            self.maintenance -= 50  # 조정된 유지 비용
        if action == 2 and self.capital > self.ITS_cost:
            self.capital -= self.ITS_cost
            self.num_ITS += 1
            self.maintenance -= 30  # 조정된 유지 비용
        if action == 3 and self.capital > self.hospitals_cost:
            self.capital -= self.hospitals_cost
            self.num_hospitals += 1
            self.maintenance -= 150  # 조정된 유지 비용
        
        # 행복도 계산
        self.calculate_happiness()
        # 인구 변동률 계산
        self.adjust_population_flow()
        # 인구 변동
        self.update_population()
        
        # 재정 변동
        self.income = self.population * 12
        self.capital += (self.income + self.maintenance)
            
        reward = self.check_if_reward()
        
        return self.get_state(), reward, True, {}
    
    def calculate_happiness(self):
        if self.num_base_station * 100 > self.population:
            self.happiness += 0.1
        else:
            self.happiness -= 0.1
        
        if self.num_ITS > self.num_apartment * 1.5:
            self.happiness += 0.1
        else:
            self.happiness -= 0.1
            
        if self.num_base_station > self.num_ITS * 0.5:
            self.happiness += 0.1
        else:
            self.happiness -= 0.1
        
        if self.num_hospitals >= self.num_apartment * 0.2:
            self.happiness += 0.1
        else:
            self.happiness -= 0.1
        
        self.happiness = min(self.happiness, 100)
        
    def adjust_population_flow(self):
        base_influx_rate = 0.05  # 기본 유입률
        base_attrition_rate = 0.001  # 기본 이탈률

        if self.happiness < 40:
            self.influx_rate = base_influx_rate * 0.5
            self.attrition_rate = base_attrition_rate * 1.5
        elif 40 <= self.happiness < 60:
            self.influx_rate = base_influx_rate * 0.7
            self.attrition_rate = base_attrition_rate * 1.2
        elif 60 <= self.happiness < 80:
            self.influx_rate = base_influx_rate * 1
            self.attrition_rate = base_attrition_rate * 1
        elif 80 <= self.happiness < 100:
            self.influx_rate = base_influx_rate * 1.2
            self.attrition_rate = base_attrition_rate * 0.7
        elif self.happiness >= 100:
            self.influx_rate = base_influx_rate * 1.5
            self.attrition_rate = base_attrition_rate * 0.5
            
    def update_population(self):
        self.current_capacity_population = self.capacity_population - self.population
        self.population += float(int(self.current_capacity_population * self.influx_rate))
        
    def check_if_reward(self):
        reward_levels = [1000, 10000, 30000]
        for index, level in enumerate(reward_levels):
            if self.population > level and self.last_reward_population_level < (index + 1):
                self.last_reward_population_level = index + 1
                return index + 1  # 인구수 임계값을 처음 초과하는 경우에만 보상 반환
        return 0
        

    def get_state(self):
        state = [
            self.capital,
            self.capacity_population,
            self.population,
            self.income,
            self.attrition_rate,
            self.influx_rate,
            self.happiness,
            self.num_apartment,
            self.num_base_station,
            self.num_ITS,
            self.num_hospitals
        ]

        return state

        
    def reset(self):
        # 리셋
        self.capital = 100000  # 초기 자본
        self.maintenance = 0 # 유지 비용
        self.income = 0 # 수익
        self.capacity_population = 0  # 수용 가능 인원
        self.population = 0  # 인구수
        self.attrition_rate = 0.001  # 초기 이탈율
        self.influx_rate = 0.05  # 초기 유입률
        self.num_apartment = 0  # 주거공간 개수
        self.num_base_station = 0  # 기지국 개수
        self.num_ITS = 0  # ITS 개수
        self.num_hospitals = 0  # 병원 개수
        self.happiness = 60  # 행복도
        self.last_reward_population_level = 0

        return self.get_state()

