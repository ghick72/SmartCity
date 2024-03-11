from shapely.geometry import Polygon, Point

# 환경 설정
UNIT = 100  # 픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로

class SmartCityEnvironment():
    def __init__(self):

        # 건축물 리스트 초기화
        self.residential_areas = []
        self.commercial_areas = []
        self.industrial_areas = []
        self.hospitals = []
        self.parks = []

        # 초기 상태 설정
        self.capital = 30000  # 초기 자본
        self.population = 0  # 초기 인구
        self.last_population = 0  # 이전 스텝의 인구 수, 초기값 설정
        self.attrition_rate = 5  # 초기 이탈율
        self.attrition_rate_multiplier = 1 #  초기 이탈율 배율
        self.influx_rate_multiplier = 1  # 초기 유입률 배율
        self.num_residential_areas = 0  # 주거공간 개수
        self.num_commercial_areas = 0  # 상업공간 개수
        self.num_industrial_areas = 0  # 산업공간 개수
        self.num_hospitals = 0  # 병원 개수
        self.num_parks = 0  # 공원 개수
        self.happiness = 60  # 행복도
        self.step_count = 0
        
        # 건물 정보 및 범위 정의
        self.buildings = [[-1 for _ in range(WIDTH)] for _ in range(HEIGHT)]

        self.agent_position = [1, 1]  # 에이전트의 초기 위치 설정
        self.num_actions = 20  # 상하좌우 * 건물 유형의 개수 (4 * 5 = 20)

            
    def is_building_at(self, x, y):
        """해당 위치에 건물이 있는지 확인합니다."""
        return self.buildings[y][x] != -1
        
    def place_building(self, building_type, x, y):
        """주어진 위치에 새로운 건물을 배치합니다."""
        if not self.is_building_at(x, y):
            self.buildings[y][x] = building_type  # buildings 리스트를 업데이트합니다.
            # 건물 유형별 개수 업데이트
            if building_type == 0: 
                self.num_residential_areas += 1
            elif building_type == 1: 
                self.num_commercial_areas += 1
            elif building_type == 2: 
                self.num_industrial_areas += 1
            elif building_type == 3: 
                self.num_hospitals += 1
            elif building_type == 4: 
                self.num_parks += 1
            return True
        else:
            return False

    def step(self, action):
        self.step_count += 1
        reward = 0
        # 모든 건물이 지어졌는지 체크
        all_buildings_constructed = all(building != -1 for row in self.buildings for building in row)

        if self.step_count == 500: # 1에피소드 완료
            return self.get_state(), 0, False , {}
        
        # 행복도 20미만 조기종료
        # if self.happiness < 20:
        #     return self.get_state(), -1000, False, {}
        
        if all_buildings_constructed:
            return self.get_state(), 500, False, {}

        self.update_population()
        self.calculate_happiness()
        self.adjust_population_flow()
                
        # 수입 및 유지비용 계산
        income = self.num_residential_areas * 200 + self.num_commercial_areas * 200 + self.num_industrial_areas * 550
        maintenance = self.num_hospitals * 250 + self.num_parks * 150
        self.capital += (income - maintenance)

        # 액션을 이동 방향과 건물 유형으로 변환
        direction = action // 5  # 이동 방향
        building_type = action % 5  # 건물 유형
        
        # 이동 방향에 따른 에이전트 위치 업데이트 로직...
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 상, 하, 좌, 우
        move = moves[direction]
        
        new_x = self.agent_position[0] + move[0]
        new_y = self.agent_position[1] + move[1]

        # 새 위치가 유효한지 확인하고, 가능하다면 이동 및 건물 배치 시도
      # 1. 이동할 수 있는 좌표인지 확인
        if 0 <= new_x < WIDTH and 0 <= new_y < HEIGHT:
            # 1-1. 이동할 수 있는 좌표라면 좌표 이동
            self.agent_position = [new_x, new_y]

            # 2. 건물을 지을 수 있는 좌표인지 확인
            if not self.is_building_at(new_x, new_y):
                # 건물 배치 비용 조회
                building_costs = [6000, 5000, 2500, 6500, 3500]  # 주거, 상업, 산업, 병원, 공원
                cost = building_costs[building_type]

                # 2-1. 건물을 지을 수 있는 좌표라면 비용 차감
                if self.capital >= cost:
                    self.capital -= cost  # 비용 차감
                    # reward += 1
                    self.place_building(building_type, new_x, new_y)  # 건물 배치
                # else:
                #     # 자본 부족으로 건물을 배치할 수 없는 경우
                #     reward -= 1
            # else:
            #     # 2-2. 건물을 지을 수 없는 좌표라면 비용 차감x
            #     return self.get_state(), 0, True, {}
        # else:
            # 1-2. 이동할 수 없는 좌표라면 좌표 이동x
            # return self.get_state(), -10, True, {}  # 이동할 수 없는 행동을 취한 것에 대한 패널티
            # reward -= 1
    

        # if self.capital < 0:
        #     reward -= 1
        
        # 보상 조건
        # population_increase = self.population - self.last_population
        
        if self.population > 500:
            reward += 1

        self.last_population = self.population
        
        return self.get_state(), reward, True, {}
 
    def get_state(self):
        state = [
            self.capital,
            self.population,
            self.attrition_rate,
            self.num_residential_areas,
            self.num_commercial_areas,
            self.num_industrial_areas,
            self.num_hospitals,
            self.num_parks,
            self.happiness
        ]

        # buildings 리스트를 state에 포함시킴
        buildings_state = [building_type for row in self.buildings for building_type in row]
        state.extend(buildings_state)
        return state

    
    def adjust_population_flow(self):
        if self.happiness < 20:
            pass
        elif 20 <= self.happiness < 40:
            self.influx_rate_multiplier = 0.5
            self.attrition_rate_multiplier = 2.0
        elif 40 <= self.happiness < 60:
            self.influx_rate_multiplier = 1
            self.attrition_rate_multiplier = 1.5
        elif 60 <= self.happiness < 80:
            self.influx_rate_multiplier = 3
            self.attrition_rate_multiplier = 0.5
        elif self.happiness >= 100:
            self.influx_rate_multiplier = 5.0
            self.attrition_rate_multiplier = 0.2

    def update_population(self):
        self.last_population = self.population
        
        # Assuming each residential area adds capacity for 100 people
        total_capacity = self.num_residential_areas * 100
        available_capacity = max(0, total_capacity - self.population)

        # 유입률 관련
        population_influx = min(available_capacity, available_capacity * 0.5 * self.influx_rate_multiplier)
        self.population += round(population_influx)

       # 이탈율 관련, 인구 이탈을 계산하고 반올림
        population_attrition = self.population * self.attrition_rate * 0.01 * self.attrition_rate_multiplier
        population_attrition = round(population_attrition)  # 인구 이탈을 반올림

        self.population = max(0, self.population - population_attrition)
        self.population = int(self.population)  # 인구수를 정수로 유지

    def calculate_happiness(self):
        happiness_change = 0
        # 그리드월드 내의 모든 좌표에 대해 검사
        for y in range(HEIGHT):
            for x in range(WIDTH):
                # 현재 좌표의 건물 유형 확인
                current_building = self.get_building_type_at(x, y)
                
                # 주거공간 상하좌우 4칸 내 산업공간 확인
                if current_building == 0:  # 주거공간
                    if self.check_industrial_nearby(x, y):
                        happiness_change -= 1
                        
                    # 상하좌우 4칸 내 상업공간 확인
                    if not self.check_commercial_nearby(x, y):
                        happiness_change -= 1
                    else:
                        happiness_change += 1
                        
                elif current_building == 2:  # 산업공간
                    if not self.check_commercial_nearby(x, y):
                        happiness_change -= 1
                    else:
                        happiness_change += 1
        
        # 주거공간 / 병원 수에 따른 행복도 조정
        if self.num_hospitals > 0:
            if (self.num_residential_areas // self.num_hospitals) < 5:
                happiness_change -= 2
            else:
                happiness_change += 2
        else:
            pass
    
        # 공원 수에 따른 행복도 조정
        happiness_change += self.num_parks
        
        # 행복도 업데이트
        self.happiness += happiness_change
        self.happiness = min(100, max(0, self.happiness))
    
    def get_building_type_at(self, grid_x, grid_y):
        return self.buildings[grid_y][grid_x]

    
    def check_industrial_nearby(self, x, y):
        for dx in range(-4, 5):  # x 방향 -4칸부터 4칸까지
            for dy in range(-4, 5):  # y 방향 -4칸부터 4칸까지
                if dx == 0 and dy == 0:
                    continue  # 현재 위치는 제외
                if 0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT:  # 그리드 경계를 넘어가지 않도록 검사
                    if self.get_building_type_at(x + dx, y + dy) == 2:  # 산업공간 확인
                        return True
        return False
    
    def check_commercial_nearby(self, x, y):
        for dx in range(-4, 5):  # x 방향 -4칸부터 4칸까지
            for dy in range(-4, 5):  # y 방향 -4칸부터 4칸까지
                if dx == 0 and dy == 0:
                    continue  # 현재 위치는 제외
                if 0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT:  # 그리드 경계를 넘어가지 않도록 검사
                    if self.get_building_type_at(x + dx, y + dy) == 1:  # 상업공간 확인
                        return True
        return False

    
    def reset(self):
        # 환경을 초기 상태로 리셋
        self.capital = 30000
        self.population = 0
        self.attrition_rate = 5
        self.num_residential_areas = 0
        self.num_commercial_areas = 0
        self.num_industrial_areas = 0
        self.num_hospitals = 0
        self.num_parks = 0
        self.happiness = 60
        self.step_count = 0

        # 건물 리스트와 행동 공간 초기화
        self.buildings = [[-1 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.agent_position = [1, 1]  # 에이전트의 초기 위치 설정

        return self.get_state()

