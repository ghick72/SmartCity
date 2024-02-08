import tkinter as tk
from shapely.geometry import Polygon, Point

# 환경 설정
UNIT = 10  # 픽셀 수
HEIGHT = 100  # 그리드 세로
WIDTH = 100  # 그리드 가로
mountain_points = [(0, 0), (0, 30*UNIT), (1*UNIT, 30*UNIT), (1*UNIT, 30*UNIT), (2*UNIT, 30*UNIT),
                   (2*UNIT, 29*UNIT), (3*UNIT, 29*UNIT), (4*UNIT, 29*UNIT), (5*UNIT, 29*UNIT),
                   (5*UNIT, 27*UNIT), (6*UNIT, 27*UNIT), (7*UNIT, 27*UNIT), (7*UNIT, 25*UNIT)
                   , (8*UNIT, 25*UNIT), (9*UNIT, 25*UNIT), (9*UNIT, 23*UNIT), (10*UNIT, 23*UNIT)
                   , (11*UNIT, 23*UNIT), (11*UNIT, 21*UNIT), (12*UNIT, 21*UNIT), (13*UNIT, 21*UNIT)
                   , (13*UNIT, 19*UNIT), (14*UNIT, 19*UNIT), (15*UNIT, 19*UNIT), (16*UNIT, 19*UNIT), 
                   (16*UNIT, 17*UNIT), (17*UNIT, 17*UNIT), (17*UNIT, 15*UNIT), (18*UNIT, 15*UNIT),
                    (19*UNIT, 15*UNIT), (19*UNIT, 13*UNIT), (20*UNIT, 13*UNIT), (21*UNIT, 13*UNIT),
                    (21*UNIT, 11*UNIT), (22*UNIT, 11*UNIT), (23*UNIT, 11*UNIT), (23*UNIT, 9*UNIT),
                    (24*UNIT, 9*UNIT), (25*UNIT, 9*UNIT), (25*UNIT, 7*UNIT), (26*UNIT, 7*UNIT),
                    (27*UNIT, 7*UNIT), (27*UNIT, 5*UNIT), (28*UNIT, 5*UNIT), (29*UNIT, 5*UNIT),
                    (29*UNIT, 3*UNIT), (30*UNIT, 3*UNIT), (31*UNIT, 3*UNIT), (31*UNIT, 1*UNIT),
                    (32*UNIT, 1*UNIT), (33*UNIT, 1*UNIT), (34*UNIT, 1*UNIT), (34*UNIT, 0), (40*UNIT, 0)
]

river_points = [
                (0, 80*UNIT), (10*UNIT, 80*UNIT), (10*UNIT,75*UNIT), (25*UNIT,75*UNIT), (25*UNIT,70*UNIT),
                (40*UNIT,70*UNIT), (40*UNIT,55*UNIT), (60*UNIT,55*UNIT), (60*UNIT,45*UNIT), (80*UNIT,45*UNIT),
                (80*UNIT,35*UNIT), (100*UNIT,35*UNIT), (100*UNIT,25*UNIT), (70*UNIT,25*UNIT), (70*UNIT,35*UNIT),
                (50*UNIT,35*UNIT), (50*UNIT,50*UNIT), (30*UNIT,50*UNIT), (30*UNIT,60*UNIT), (15*UNIT,60*UNIT),
                (15*UNIT,70*UNIT), (0,70*UNIT)
                ]

class SmartCityEnvironment:
    def __init__(self, window):
        self.window = window
        self.window.title("Smart City Simulation")
        self.canvas = tk.Canvas(window, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        self.draw_grid()
        self.draw_natural_features()
        self.calculate_buildable_area()
        self.canvas.pack()

        # 건축물 리스트 초기화
        self.base_stations = []
        self.residential_areas = []
        self.commercial_areas = []
        self.industrial_areas = []
        self.hospitals = []
        self.parks = []

        # 선택 단계 초기화
        self.current_action_phase = 'select_building_type'  # 'select_building_type' 또는 'select_location'
        self.selected_building_type = None
        self.selected_location = None

        # 초기 상태 설정
        self.capital = 50000  # 초기 자본
        self.population = 0  # 초기 인구
        self.attrition_rate = 0.01  # 초기 이탈율
        self.num_base_stations = 0  # 기지국 개수
        self.num_residential_areas = 0  # 주거공간 개수
        self.num_commercial_areas = 0  # 상업공간 개수
        self.num_industrial_areas = 0  # 산업공간 개수
        self.num_hospitals = 0  # 병원 개수
        self.num_parks = 0  # 공원 개수
        self.happiness = 60  # 행복도
        self.step_count = 0
        
        # 건물 정보 및 범위 정의
        self.buildings = []  # (type, x, y) 형식의 튜플 리스트
        self.building_range = {
            "base_station": (7, 7),  # 중간 범위
            "residential": (0, 0),  # 영향 범위 없음, 위치 정보만 필요
            "commercial": (5, 5),  # 좁은 범위
            "industrial": (5, 5),  # 좁은 범위
            "hospital": (7, 7),  # 중간 범위
            "park": (11, 11)  # 넓은 범위
        }

    def draw_grid(self):
        for c in range(0, WIDTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, HEIGHT * UNIT)
        for r in range(0, HEIGHT * UNIT, UNIT):
            self.canvas.create_line(0, r, WIDTH * UNIT, r)

    def draw_natural_features(self):
        self.canvas.create_polygon(mountain_points, fill="green")
        self.canvas.create_polygon(river_points, fill="blue")

    def calculate_buildable_area(self):
        self.can_build = []
        river_polygon = Polygon(river_points)
        mountain_polygon = Polygon(mountain_points)
        for x in range(0, WIDTH * UNIT, UNIT):
            for y in range(0, HEIGHT * UNIT, UNIT):
                cell_center = Point(x + UNIT / 2, y + UNIT / 2)
                if not river_polygon.contains(cell_center) and not mountain_polygon.contains(cell_center):
                    self.can_build.append((x, y))

    def place_building(self, location_index):
        # 잘못된 위치를 선택한 경우 바로 다음 스텝으로 넘어가도록 설정
        if 0 <= location_index < len(self.can_build):
            grid_x, grid_y = self.can_build.pop(location_index)  # 선택한 위치 삭제
            building_type = self.selected_building_type

            # 선택된 건축물 유형에 따라 리스트에 위치 추가
            building_list = self.get_building_list(building_type)
            if building_list is not None:
                building_list.append((grid_x, grid_y))

            # 캔버스 업데이트
            self.update_canvas()

            # 건물을 성공적으로 지었을 때는 다음 스텝으로 넘어가도록 설정
            self.current_action_phase = 'select_building_type'
            return True, 0, False  # 성공적으로 건물을 지었지만, 추가 보상은 없음
        else:
            # 잘못된 위치를 선택했을 경우, 다음 스텝으로 넘어갈 수 있도록 설정
            self.current_action_phase = 'select_building_type'
            return False, 0, False  # 잘못된 위치 선택, 보상 없음, 게임은 종료되지 않음

    def get_building_list(self, building_type):
        if building_type == 0:
            return self.base_stations
        elif building_type == 1:
            return self.residential_areas
        elif building_type == 2:
            return self.commercial_areas
        elif building_type == 3:
            return self.industrial_areas
        elif building_type == 4:
            return self.hospitals
        elif building_type == 5:
            return self.parks
        else:
            return None

    def update_canvas(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_natural_features()
        # 각 건축물 리스트를 순회하며 캔버스에 그림
        self.draw_buildings(self.base_stations, "blue")
        self.draw_buildings(self.residential_areas, "purple")
        self.draw_buildings(self.commercial_areas, "black")
        self.draw_buildings(self.industrial_areas, "yellow")
        self.draw_buildings(self.hospitals, "red")
        self.draw_buildings(self.parks, "gray")

    def draw_buildings(self, buildings, color):
        for x, y in buildings:
            self.canvas.create_rectangle(x*UNIT, y*UNIT, (x+1)*UNIT, (y+1)*UNIT, fill=color)

    def step(self, action):
        # Define the cost and effects of each action
        actions = {
            0: {"name": "build_base_station", "cost": 1200, "steps": 1, "maintenance": 20},
            1: {"name": "build_residential", "cost": 6000, "steps": 4, "capacity": 100},
            2: {"name": "build_commercial", "cost": 5000, "steps": 4, "income_per_step": 100},
            3: {"name": "build_industrial", "cost": 2500, "steps": 4, "income_per_step": 300},
            4: {"name": "build_hospital", "cost": 6500, "steps": 1, "maintenance": 240},
            5: {"name": "build_park", "cost": 450, "steps": 1, "maintenance": 40}
        }

        self.step_count += 1

        action_details = actions.get(action, {})
        cost = action_details.get("cost", 0)
        steps_required = action_details.get("steps", 0)

        # Initialize done to False
        done = False

        if self.step_count == 21900:
            return self.get_state(), -1, True

        # Check if there is enough capital to perform the action
        if self.capital < cost:
            return self.get_state(), -1, False  # 스탭 넘어가기

        # Deduct the cost and perform the action
        self.capital -= cost
        self.steps_taken += steps_required  # Simulate time passing for construction

        if self.current_action_phase == 'select_building_type':
            self.selected_building_type = action
            self.current_action_phase = 'select_location'
            return self.get_state(), 0, False, {}
        elif self.current_action_phase == 'select_location':
            success, reward, done = self.place_building(action)
            if success:
                self.current_action_phase = 'select_building_type'

        # Implement action effects here, similar to the provided snippet

        # Update population and calculate happiness
        self.update_population()
        self.calculate_happiness()
        # Adjust population influx and attrition based on happiness
        self.adjust_population_flow()

        # Check for early termination conditions based on happiness
        if self.happiness < 20:
            return self.get_state(), -1, True

        # Existing implementation of reward calculation and returning the state, reward, done, and info

        # Check for early termination conditions
        if self.capital < 0 and self.negative_income_steps >= 8:
            done = True
        if self.steps_since_last_population_check >= 1000 and self.population <= self.last_population_check:
            done = True


        # 보상 조건 수정
        population_increase = self.population - self.last_population

        reward = 0
        
        if population_increase > 0:
            if population_increase >= 10000:
                reward += 20
            elif population_increase >= 1000:
                reward += 10
            elif population_increase >= 100:
                reward += 5
            elif population_increase >= 20:
                reward += 1

        self.last_population = self.population

        return self.get_state(), reward, done, {}
 
    def get_state(self):
        state = [
            self.capital, self.population, self.attrition_rate,
                self.num_base_stations, self.num_residential_areas,
                self.num_commercial_areas, self.num_industrial_areas,
                self.num_hospitals, self.num_parks, self.happiness
        ]
        
        # Convert building locations into a normalized format
        buildings_state = []
        for building in self.buildings:
            building_type, x, y = building
            # Normalize x and y to the grid size
            normalized_x = x / (WIDTH * UNIT)
            normalized_y = y / (HEIGHT * UNIT)
            buildings_state.append((building_type, normalized_x, normalized_y))
        
        # Normalize can_build locations
        can_build_state = [(x / (WIDTH * UNIT), y / (HEIGHT * UNIT)) for x, y in self.can_build]

        # Append buildings_state and can_build_state to the existing state
        state.append(buildings_state)
        state.append(can_build_state)

        return state

    
    def calculate_distance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def find_buildings_within_range(self, building_type, x, y, range_x, range_y):
        in_range_buildings = []
        for b_type, b_x, b_y in self.buildings:
            if b_type == building_type and self.calculate_distance(x, y, b_x, b_y) <= max(range_x, range_y) * UNIT / 2:
                in_range_buildings.append((b_type, b_x, b_y))
        return in_range_buildings

    def calculate_attrition_rate_decrease(self):
        total_residential_areas = len(self.residential_areas)
        if total_residential_areas == 0:
            return  # 주거 공간이 없으면 함수를 종료
    
        # 병원 범위 내 주거 공간 계산
        covered_residential_areas = 0
        for hospital_x, hospital_y in self.hospitals:
            # 병원 범위 설정 (중간 범위)
            hospital_range_x, hospital_range_y = self.building_range["hospital"]
            for residential_x, residential_y in self.residential_areas:
                distance = self.calculate_distance(hospital_x, hospital_y, residential_x, residential_y)
                if distance <= max(hospital_range_x, hospital_range_y) * UNIT / 2:
                    covered_residential_areas += 1
    
        # 전체 주거 공간 대비 병원 범위 내 주거 공간의 비율 계산
        coverage_ratio = covered_residential_areas / total_residential_areas
    
        # 이탈율 감소 계산 (비율의 절반만큼 이탈율 감소)
        attrition_rate_decrease = coverage_ratio / 2
        self.attrition_rate *= (1 - attrition_rate_decrease)
    
        # 이탈율이 0 미만으로 떨어지지 않도록 보장
        self.attrition_rate = max(self.attrition_rate, 0)
    
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
        population_influx = min(available_capacity, available_capacity * 0.2 * self.influx_rate_multiplier)
        self.population += round(population_influx)

       # 이탈율 관련, 인구 이탈을 계산하고 반올림
        self.calculate_attrition_rate_decrease()
        population_attrition = self.population * self.attrition_rate * self.attrition_rate_multiplier
        population_attrition = round(population_attrition)  # 인구 이탈을 반올림

        self.population = max(0, self.population - population_attrition)
        self.population = int(self.population)  # 인구수를 정수로 유지


    def calculate_happiness(self):
        happiness_change = 0

        # 기지국 범위 내 건축물 처리
        base_stations = [(x, y) for b, x, y in self.buildings if b == "base_station"]
        for x, y in base_stations:
            buildings_in_range = self.find_buildings_within_range("base_station", x, y, 7, 7, self.buildings)
            if len(buildings_in_range) > 32:
                happiness_change -= (len(buildings_in_range) - 32) * 0.1
            elif len(buildings_in_range) < 16:
                happiness_change += (16 - len(buildings_in_range)) * 0.1

        # 주거 공간 근처의 상업 공간 처리
        residential_areas = [(x, y) for b, x, y in self.buildings if b == "residential"]
        for x, y in residential_areas:
            commercial_in_range = self.find_buildings_within_range("commercial", x, y, 5, 5, self.buildings)
            if len(commercial_in_range) > 2:
                happiness_change -= 0.05
            elif len(commercial_in_range) == 0:
                happiness_change -= 0.05

        # 산업 공간 근처의 상업 공간 처리
        industrial_areas = [(x, y) for b, x, y in self.buildings if b == "industrial"]
        for x, y in industrial_areas:
            commercial_in_range = self.find_buildings_within_range("commercial", x, y, 5, 5, self.buildings)
            if len(commercial_in_range) == 0:
                happiness_change -= 0.05

        # 공원의 영향 범위 처리
        for x, y in residential_areas + industrial_areas:
            park_in_range = self.find_buildings_within_range("park", x, y, 11, 11, self.buildings)
            if park_in_range:
                happiness_change += 0.1

        # 행복도 업데이트
        self.happiness += happiness_change
        self.happiness = min(100, max(0, self.happiness))
    
    def reset(self):
        # 환경을 초기 상태로 리셋
        self.capital = 50000
        self.population = 0
        self.attrition_rate = 0.01
        self.num_base_stations = 0
        self.num_residential_areas = 0
        self.num_commercial_areas = 0
        self.num_industrial_areas = 0
        self.num_hospitals = 0
        self.num_parks = 0
        self.happiness = 60
        return self.get_state()
    
# 애플리케이션 실행
window = tk.Tk()
app = SmartCityEnvironment(window)
window.mainloop()
