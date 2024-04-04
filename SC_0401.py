class SmartCityEnvironment():
    def __init__(self):
        # 초기 상태
        self.capital = 500  # 초기 자본
        self.maintenance = 0 # 유지 비용
        self.income = 0 # 수익

        self.capacity_population = 0  # 수용 가능 인원
        self.population = 0  # 초기 인구수
        self.current_capacity_population = 0  # 수용 가능 인구 중 수용 가능 인구
        self.pop_size = 100
        
        self.influx_rate = 0.1  # 초기 유입률
        self.attrition_rate = 0.01  # 초기 이탈율

        self.base_influx_rate = 0.1  # 기본 유입률
        self.base_attrition_rate = 0.01  # 기본 이탈률

        self.num_apartment = 0  # 주거공간 개수
        self.num_base_station = 0  # 기지국 개수
        self.num_ITS = 0  # ITS 개수
        self.num_hospitals = 0  # 병원 개수

        self.happiness = 0  # 행복도
        self.bench_happiness = 90  # 기준 행복도

        self.last_reward_population_level = 0

        self.step_count = 0

        self.check_hos = 0  # 한 번만 업데이트 해주기 위해
        self.check_ITS = 0
        
    def step(self, action):
        self.done = False

        # 보상 초기화
        reward = 0

        self.apartment_cost = 60
        self.base_station_cost = 50
        self.ITS_cost = 25
        self.hospitals_cost = 65


        if action == 0 and self.capital > self.apartment_cost:
            self.capital -= self.apartment_cost
            self.num_apartment += 1
            self.capacity_population += self.pop_size
        if action == 1 and self.capital > self.base_station_cost:
            self.capital -= self.base_station_cost
            self.num_base_station += 1
            self.maintenance -= 2   # 유지비용
        if action == 2 and self.capital > self.ITS_cost:
            self.capital -= self.ITS_cost
            self.num_ITS += 1
            self.maintenance -= 1   # 유지비용
        if action == 3 and self.capital > self.hospitals_cost:
            self.capital -= self.hospitals_cost
            self.num_hospitals += 1
            self.maintenance -= 6   # 유지비용
        
        # 행복도 계산
        try:
            self.calculate_happiness()
        except ZeroDivisionError:
            pass

        # 인구 변동률 계산
        self.adjust_population_flow()
        # 인구 변동
        self.update_population()
        
        # 재정 변동
        self.income = self.population * 10
        self.capital += self.income + self.maintenance
        
        # reward = self.check_if_reward()

        reward = (self.population // 5000) + min((self.happiness - self.bench_happiness), -10)
        ## 만족도를 도달해야 양의 보상을 받을 수 있을 것. 또한, 인구수 만명당 보상으로 단위 보상 주기

        # print(reward)
        
        self.check_done()

        # print("이건 되나")

        return self.get_state(), reward, self.done, {}
        
    # def check_if_reward(self):
    #     ######################################## 다시 확인합시다
    #     return 0
        
    def check_done(self):
        self.step_count += 1
        step_done = 0

        if self.step_count % 10 == 0:  # 10스탭마다 인구가 0명 이하이면 조기종료. 이는 학습 중간에도 유효함.
            step_done = 1
            if step_done == 1 and self.population <= 0:
                self.done = True
            else:
                step_done = 0

        if self.step_count >= 1040:
            self.done = True
        

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

    
    def adjust_population_flow(self):
        if (self.num_hospitals * 500 >= self.population) and (self.check_hos == 0):
            self.attrition_rate = self.base_attrition_rate - 0.005
            self.check_hos = 1
        elif (self.num_hospitals * 500 < self.population) and (self.check_hos == 1):  # 다시 미달이되면 초기화
            self.attrition_rate = self.base_attrition_rate
            self.check_hos = 0
        
        if (self.num_ITS * 400 >= self.population) and (self.check_ITS == 0):
            self.influx_rate = self.base_influx_rate + 0.1
            self.check_ITS = 1
        elif (self.num_ITS * 400 < self.population) and (self.check_ITS == 1):  # 다시 미달이되면 초기화
            self.influx_rate = self.base_influx_rate
            self.check_ITS = 0

        if self.check_ITS and self.check_hos == 1:
            self.pop_size = 200
        else:
            self.pop_size = 100


    def update_population(self):
        self.current_capacity_population = self.capacity_population - self.population
        self.population += float(int(self.current_capacity_population * self.influx_rate))
        
    def calculate_happiness(self):
        self.happiness = max(0, min(300 - (self.population // self.num_base_station), 100))
        ## 인구수 200명당 기지국 1개 일 때 만족도 100, 인구수 300명당 기지국 1개일 때 만족도 0


    def reset(self):
        # 리셋
        self.capital = 300  # 초기 자본
        self.maintenance = 0 # 유지 비용
        self.income = 0 # 수익

        self.capacity_population = 0  # 수용 가능 인원
        self.population = 0  # 인구수
        self.current_capacity_population = 0  # 수용 가능 인구 중 수용 가능 인구
        self.pop_size = 100
        
        self.influx_rate = 0.1  # 초기 유입률
        self.attrition_rate = 0.01  # 초기 이탈율

        self.base_influx_rate = 0.1  # 기본 유입률
        self.base_attrition_rate = 0.01  # 기본 이탈률

        self.num_apartment = 0  # 주거공간 개수
        self.num_base_station = 0  # 기지국 개수
        self.num_ITS = 0  # ITS 개수
        self.num_hospitals = 0  # 병원 개수

        self.happiness = 0  # 행복도
        self.bench_happiness = 90  # 기준 행복도

        self.last_reward_population_level = 0

        self.step_count = 0

        self.check_hos = 0  # 한 번만 업데이트 해주기 위해
        self.check_ITS = 0

        return self.get_state()
 