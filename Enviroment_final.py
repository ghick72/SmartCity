class CityEnviroment:
    def __init__(self):
        self.initial_apartments = 5
        self.initial_hospitals = 3
        self.initial_communication_towers = 5
        self.initial_ITS = 2
        self.apartments = self.initial_apartments
        self.hospitals = self.initial_hospitals
        self.communication_towers = self.initial_communication_towers
        self.ITS = self.initial_ITS
        self.population = 1000
        self.capital = 10000000
        self.satisfaction = 20
        self.last_month_satisfaction = 20
        self.week = 0
        self.consecutive_increases = 0
        self.population_decay_base = 0.01
        self.monthly_population = [self.population] * 4
        self.last_month_population = [0] * 4
        self.monthly_capital = [self.capital] * 4
        self.last_month_capital = [0] * 4


    def step(self, action):
        # 건물/시스템의 건설/확장 비용
        apartment_cost = 2400000
        hospital_cost = 35000000
        communication_tower_cost = 20000
        ITS_cost = 3000000

        # 행동에 따른 처리
        if action == 1 and self.capital >= apartment_cost:  # 아파트 건설
            self.apartments += 1
            self.population += int(100 * (1.03 ** max(0, self.ITS - self.initial_ITS)))  # ITS 확장에 따른 인구 증가율 적용
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

        # 아파트 철거 조건 적용
        if self.population <= (self.apartments / 2) * 100:
            demolished_apartments = round(self.apartments * 0.1)
            self.apartments = max(0, self.apartments - demolished_apartments)

        # 만족도 계산
        if self.satisfaction < 20:
            if 200 >= self.population // self.communication_towers:
                self.satisfaction += 1
            else:
                self.satisfaction -= 1

            if self.communication_towers >= self.ITS:
                self.satisfaction += 1
            else:
                self.satisfaction -= 1
        elif self.satisfaction < 100:
            if 200 >= self.population // self.communication_towers:
                self.satisfaction += 0.5
            else:
                self.satisfaction -= 1

            if self.communication_towers >= self.ITS:
                self.satisfaction += 0.5
            else:
                self.satisfaction -= 1
        elif self.satisfaction >= 100:
            if 200 < self.population // self.communication_towers:
                self.satisfaction -= 1

            if self.communication_towers < self.ITS:
                self.satisfaction -= 1

        # 보상 계산
        reward = 0

        if self.week % 4 == 0:  # 매 4주마다 보상 계산
            current_month_avg_population = sum(self.monthly_population) / 4
            last_month_avg_population = sum(self.last_month_population) / 4
            current_month_avg_capital = sum(self.monthly_capital) / 4
            last_month_avg_capital = sum(self.last_month_capital) / 4

            # 인구와 자본이 증가했는지 확인
            population_increase = current_month_avg_population > last_month_avg_population
            capital_increase = current_month_avg_capital > last_month_avg_capital
            


            # 이번달의 상태를 지난달로 업데이트
            self.last_month_population = self.monthly_population.copy()
            self.last_month_capital = self.monthly_capital.copy()
            self.monthly_population = [self.population] * 4
            self.monthly_capital = [self.capital] * 4
        else:
            # 매주 인구 및 자본 기록
            self.monthly_population[self.week % 4] = self.population
            self.monthly_capital[self.week % 4] = self.capital

        # 인구 및 자본 업데이트
        self.update_population()
        self.update_capital()

        self.last_month_satisfaction = self.satisfaction

        # 시뮬레이션 종료 조건 확인
        self.week += 1
        done = self.week >= 1040 or self.population < 100 or self.satisfaction < 0

        # 다음 상태 반환
        next_state = [self.apartments, self.hospitals, self.communication_towers, self.ITS, 
                      self.population, self.capital,  self.satisfaction]  # self.week, self.consecutive_increases,
        return next_state, reward, done

    def update_population(self):
        # 인구 감소 확률 계산
        pop_decay_rate = self.population_decay_base
        if self.apartments > (self.hospitals + self.ITS) * 2:
            pop_decay_rate *= 1.1
        pop_decay_rate *= (0.98 ** self.hospitals)

        # 인구 감소 적용
        self.population -= int(self.population * pop_decay_rate)
        self.population = max(self.population, 0)

    def update_capital(self):
        # 통신 기지국에 의한 수익
        if self.population < 1000:
            self.capital += 900000 * self.communication_towers
        else:
            self.capital += (1000000 + 100000 * (self.population // 1000 - 1)) * self.communication_towers

        # ITS 유지비
        self.capital -= self.ITS * 90000

        self.capital = int(max(self.capital, 0))

    def reset(self):
        # 모든 변수 초기화
        self.apartments = self.initial_apartments
        self.hospitals = self.initial_hospitals
        self.communication_towers = self.initial_communication_towers
        self.ITS = self.initial_ITS
        self.population = 1000
        self.capital = 10000000
        self.satisfaction = 20
        self.week = 0
        self.consecutive_increases = 0
        self.last_reward_population = 1000
        self.last_reward_capital = 10000000
        self.population_decay_base = 0.01
        self.monthly_population = [self.population] * 4
        self.last_month_population = [0] * 4
        self.monthly_capital = [self.capital] * 4
        self.last_month_capital = [0] * 4

        return [self.apartments, self.hospitals, self.communication_towers, self.ITS, 
                      self.population, self.capital,  self.satisfaction]  # self.week, self.consecutive_increases,
