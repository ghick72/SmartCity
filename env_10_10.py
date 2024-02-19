import tkinter as tk
from shapely.geometry import Polygon, Point

# 환경 설정
UNIT = 100  # 픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로

class SmartCityEnvironment:
    def __init__(self, window):
        self.window = window
        self.window.title("Smart City Simulation")
        self.canvas = tk.Canvas(window, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
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
        self.attrition_rate = 0.05  # 초기 이탈율
        self.num_residential_areas = 0  # 주거공간 개수
        self.num_commercial_areas = 0  # 상업공간 개수
        self.num_industrial_areas = 0  # 산업공간 개수
        self.num_hospitals = 0  # 병원 개수
        self.num_parks = 0  # 공원 개수
        self.happiness = 60  # 행복도
        self.step_count = 0
        
        # 건물 정보 및 범위 정의
        self.buildings = []  # (type, x, y) 형식의 튜플 리스트

        # 행동 공간 초기화
        self.actions = []

        # 그리드 사이즈 및 건물 유형
        grid_size = (WIDTH, HEIGHT)
        building_types = 5  # 주거공간(0), 상업공간(1), 산업공간(2), 병원(3), 공원(4)

        # 각 좌표마다 가능한 모든 건물 유형에 대한 행동 생성
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for building_type in range(building_types):
                    self.actions.append((x, y, building_type))

    def draw_grid(self):
        for c in range(0, WIDTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, HEIGHT * UNIT)
        for r in range(0, HEIGHT * UNIT, UNIT):
            self.canvas.create_line(0, r, WIDTH * UNIT, r)
    def is_building_at(self, x, y):
        for _, bx, by in self.buildings:
            if bx == x * UNIT and by == y * UNIT:
                return True  # 해당 위치에 건물이 있음
        return False  # 해당 위치에 건물이 없음

    def place_building(self, x, y, building_type):
        """주어진 위치에 새로운 건물을 배치합니다."""
        if not self.is_building_at(x, y):
            self.buildings.append((building_type, x * UNIT, y * UNIT))
            if building_type == 0: self.num_residential_areas += 1
            elif building_type == 1: self.num_commercial_areas += 1
            elif building_type == 2: self.num_industrial_areas += 1
            elif building_type == 3: self.num_hospitals += 1
            elif building_type == 4: self.num_parks += 1
            self.update_canvas()
            return True  # 배치 성공
        else:
            return False  # 배치 실패, 이미 건물이 존재함


    def update_canvas(self):
        self.canvas.delete("all")  # 캔버스 초기화
        self.draw_grid()  # 그리드 다시 그리기

        # 건물 유형별 색상 정의
        colors = {
            0: "purple",  # 주거
            1: "black",   # 상업
            2: "yellow",  # 산업
            3: "red",     # 병원
            4: "green",   # 공원
            -1: "white"   # 건물 없음
        }

        # 모든 건물을 순회하며 캔버스에 그림
        for building in self.buildings:
            building_type, x, y = building
            color = colors.get(building_type, "white")  # 건물 유형에 따른 색상 선택, 기본값은 'white'
            # 그리드 좌표를 픽셀 좌표로 변환
            pixel_x, pixel_y = x * UNIT, y * UNIT
            self.canvas.create_rectangle(pixel_x, pixel_y, pixel_x + UNIT, pixel_y + UNIT, fill=color)

    def step(self, action_index):
        self.step_count += 1
        
        if self.step_count % 10 == 0:
            action = self.actions[action_index]
            x, y, building_type = action

            # 건물 배치 비용 조회
            building_costs = [6000, 5000, 2500, 6500, 450]  # 주거, 상업, 산업, 병원, 공원
            cost = building_costs[building_type]

            if self.capital >= cost:
                self.capital -= cost  # 비용 차감
                placed = self.place_building(x, y, building_type)  # 건물 배치 성공 여부 반환
                if not placed:
                    # 이미 건물이 존재하는 경우, 다음 스텝으로 넘어감
                    return self.get_state(), 0, False
            else:
                # 자본 부족으로 건물을 배치할 수 없는 경우
                return self.get_state(), 0, False 
        
        # 수입 및 유지비용 계산
        income = self.num_residential_areas * 25 + self.num_commercial_areas * 25 + self.num_industrial_areas * 75
        maintenance = self.num_hospitals * 100 + self.num_parks * 150
        self.capital += (income - maintenance)
        
        # Initialize done to False
        done = False

        if self.step_count == 21900: # 1에피소드 완료
            return self.get_state(), -1, True

        # Update population and calculate happiness
        self.update_population()
        self.calculate_happiness()
        # Adjust population influx and attrition based on happiness
        self.adjust_population_flow()

        # Check for early termination conditions based on happiness
        if self.happiness < 20:
            return self.get_state(), -100, True


        # # Check for early termination conditions
        # if self.capital < 0 and self.negative_income_steps >= 8:
        #     done = True
        # if self.steps_since_last_population_check >= 1000 and self.population <= self.last_population_check:
        #     done = True


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
            self.capital, self.population, self.attrition_rate,self.num_residential_areas,
            self.num_commercial_areas, self.num_industrial_areas,
            self.num_hospitals, self.num_parks, self.happiness
        ]

        # 모든 좌표에 대한 건물 상태를 초기화
        buildings_state = [[-1 for _ in range(WIDTH)] for _ in range(HEIGHT)]

        # 건물이 위치한 좌표에 대해 건물 유형 할당
        for building in self.buildings:
            building_type, x, y = building
            grid_x, grid_y = x // UNIT, y // UNIT  # 그리드 좌표로 변환
            buildings_state[grid_y][grid_x] = building_type

        # buildings_state를 state에 추가
        for row in buildings_state:
            for building_type in row:
                state.append(building_type)

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
        else:
            # Default values if needed
            self.influx_rate_multiplier = 1
            self.attrition_rate_multiplier = 1

    def update_population(self):
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
        if self.num_residential_areas < 5 or self.num_hospitals < 5:
            happiness_change -= 1
        else:
            happiness_change += 1
    
        # 공원 수에 따른 행복도 조정
        happiness_change += self.num_parks
        
        # 행복도 업데이트
        self.happiness += happiness_change
        self.happiness = min(100, max(0, self.happiness))
    
    def get_building_type_at(self, grid_x, grid_y):
        """주어진 그리드 위치에 있는 건물의 유형을 반환합니다."""
        for building_type, x, y in self.buildings:
            # 저장된 건물 위치를 그리드 단위로 변환하여 비교
            if x // UNIT == grid_x and y // UNIT == grid_y:
                return building_type
        return -1  # 해당 위치에 건물이 없는 경우

    
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
        self.buildings = []
        self.actions = []

        # 가능한 모든 건물 배치 액션을 다시 생성
        for x in range(WIDTH):
            for y in range(HEIGHT):
                for building_type in range(5):  # 건물 유형의 개수
                    self.actions.append((x, y, building_type))

        # 캔버스와 관련된 추가적인 초기화 작업이 필요하다면 여기서 수행
        self.update_canvas()

        return self.get_state()

    
# 애플리케이션 실행
window = tk.Tk()
app = SmartCityEnvironment(window)
window.mainloop()
