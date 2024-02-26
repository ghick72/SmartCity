import tkinter as tk
from shapely.geometry import Polygon, Point
import time
from random import choice

# 환경 설정
UNIT = 100  # 픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로

class SmartCityEnvironment(tk.Tk):
    def __init__(self, render_speed=0.01):
        super().__init__()  # tk.Tk 클래스의 __init__ 호출
        self.render_speed=render_speed
        self.title("Smart City Simulation")
        self.canvas = tk.Canvas(self, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        self.draw_grid()
        self.canvas.pack()

        # 건축물 리스트 초기화
        self.residential_areas = []
        self.commercial_areas = []
        self.industrial_areas = []
        self.hospitals = []
        self.parks = []

        # 초기 상태 설정
        self.capital = 20000  # 초기 자본
        self.population = 0  # 초기 인구
        self.last_population = 0  # 이전 스텝의 인구 수, 초기값 설정
        self.attrition_rate = 0.05  # 초기 이탈율
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

        # 행동 공간 초기화
        self.actions = []

        # 그리드 사이즈 및 건물 유형
        grid_size = (WIDTH, HEIGHT)
        building_types = 5  # 주거공간(0), 상업공간(1), 산업공간(2), 병원(3), 공원(4)

        # 각 좌표마다 가능한 모든 건물 유형에 대한 행동 생성
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for building_type in range(building_types):
                    self.actions.append((building_type, x, y))
        
    def draw_grid(self):
        for c in range(0, WIDTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, HEIGHT * UNIT)
        for r in range(0, HEIGHT * UNIT, UNIT):
            self.canvas.create_line(0, r, WIDTH * UNIT, r)
            
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
            self.update_canvas()  # 캔버스를 업데이트하여 변경사항을 반영합니다.
            return True
        else:
            return False

    def update_canvas(self):
        self.canvas.delete("all")  # 캔버스에 그려진 모든 것을 지웁니다.
        self.draw_grid()  # 그리드를 다시 그립니다.
        # HEIGHT x WIDTH 그리드를 순회하며 각 위치에 대한 처리를 수행합니다.
        for y in range(HEIGHT):
            for x in range(WIDTH):
                building_type = self.buildings[y][x]  # 해당 위치의 건물 유형을 가져옵니다.
                # 건물 유형에 따라 색상을 결정합니다. -1은 흰색, 그 외는 각 건물 유형별 색상을 사용합니다.
                colors = ["purple", "black", "yellow", "red", "green", "white"]  # -1을 위한 흰색이 추가되었습니다.
                color = colors[building_type] if building_type != -1 else colors[-1]  # 건물 유형별 색상을 선택하거나, -1일 경우 흰색을 선택합니다.
                # 해당 건물 유형에 맞는 색상으로 사각형을 그립니다.
                self.canvas.create_rectangle(x * UNIT, y * UNIT, (x + 1) * UNIT, (y + 1) * UNIT, fill=color)


    def step(self, action_index):
        self.step_count += 1
                
        if self.step_count % 5 == 0:
            action = self.actions[action_index]
            building_type, x, y = action

            # 건물 배치 비용 조회
            building_costs = [6000, 5000, 2500, 6500, 450]  # 주거, 상업, 산업, 병원, 공원
            cost = building_costs[building_type]
        
            if self.capital >= cost:
                self.capital -= cost  # 비용 차감
                placed = self.place_building(building_type, x, y)  # 건물 배치 성공 여부 반환
                if not placed:
                    # 이미 건물이 존재하는 경우, 다음 스텝으로 넘어감
                    return self.get_state(), 0, False ,{}
            else:
                # 자본 부족으로 건물을 배치할 수 없는 경우
                return self.get_state(), 0, False ,{}
        
        # 수입 및 유지비용 계산
        income = self.num_residential_areas * 25 + self.num_commercial_areas * 25 + self.num_industrial_areas * 75
        maintenance = self.num_hospitals * 100 + self.num_parks * 150
        self.capital += (income - maintenance)
        
        # Initialize done to False
        done = False

        if self.step_count == 250: # 1에피소드 완료
            return self.get_state(), -1, True , {}

        # Update population and calculate happiness
        self.update_population()
        self.calculate_happiness()
        # Adjust population influx and attrition based on happiness
        self.adjust_population_flow()

        # Check for early termination conditions based on happiness
        if self.happiness < 20:
            return self.get_state(), -100, True, {}

        # 보상 조건 수정
        population_increase = self.population - self.last_population

        reward = 0
        
        if population_increase > 0:
            if population_increase >= 100:
                reward += 20
            elif population_increase >= 50:
                reward += 10
            elif population_increase >= 20:
                reward += 5
            elif population_increase >= 10:
                reward += 1

        self.last_population = self.population

        return self.get_state(), reward, done, {}
 
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
            self.influx_rate_multiplier = 0.2
            self.attrition_rate_multiplier = 2.0
        elif 40 <= self.happiness < 60:
            self.influx_rate_multiplier = 0.5
            self.attrition_rate_multiplier = 1.5
        elif 60 <= self.happiness < 80:
            self.influx_rate_multiplier = 1.5
            self.attrition_rate_multiplier = 0.5
        elif self.happiness >= 100:
            self.influx_rate_multiplier = 2.0
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
        population_attrition = self.population * self.attrition_rate * self.attrition_rate_multiplier
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
                happiness_change -= 1
            else:
                happiness_change += 1
        else:
            pass
    
        # 공원 수에 따른 행복도 조정
        happiness_change += self.num_parks
        
        # 행복도 업데이트
        self.happiness += happiness_change
        self.happiness = min(100, max(0, self.happiness))
    
    def get_building_type_at(self, grid_x, grid_y):
        """주어진 그리드 위치에 있는 건물의 유형을 반환합니다."""
        return self.buildings[grid_y][grid_x]


    
    def check_industrial_nearby(self, x, y):
        """상하좌우 4칸 내에 산업공간이 있는지 확인합니다."""
        for dx in range(-4, 5):  # x 방향 -4칸부터 4칸까지
            for dy in range(-4, 5):  # y 방향 -4칸부터 4칸까지
                if dx == 0 and dy == 0:
                    continue  # 현재 위치는 제외
                if 0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT:  # 그리드 경계를 넘어가지 않도록 검사
                    if self.get_building_type_at(x + dx, y + dy) == 2:  # 산업공간 확인
                        return True
        return False
    
    def check_commercial_nearby(self, x, y):
        """상하좌우 4칸 내에 상업공간이 있는지 확인합니다."""
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
        self.capital = 20000
        self.population = 0
        self.attrition_rate = 0.05
        self.num_residential_areas = 0
        self.num_commercial_areas = 0
        self.num_industrial_areas = 0
        self.num_hospitals = 0
        self.num_parks = 0
        self.happiness = 60
        self.step_count = 0

        # 건물 리스트와 행동 공간 초기화
        self.buildings = [[-1 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.actions = []

        # 가능한 모든 건물 배치 액션을 다시 생성
        for x in range(WIDTH):
            for y in range(HEIGHT):
                for building_type in range(5):  # 건물 유형의 개수
                    self.actions.append((x, y, building_type))

        # 캔버스와 관련된 추가적인 초기화 작업이 필요하다면 여기서 수행
        self.update_canvas()

        return self.get_state()

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
