import tkinter as tk

# 환경 설정
UNIT = 100  # 픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로

class SmartCityEnvironment(tk.Tk):
    def __init__(self, render_speed=0.01):
        super().__init__()  # tk.Tk 클래스의 __init__ 호출
        self.render_speed = render_speed
        self.title("Smart City Simulation")
        self.geometry(f"{WIDTH*UNIT}x{HEIGHT*UNIT}")  # 창 크기 설정
        self.canvas = tk.Canvas(self, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        self.canvas.pack()
        self.actions = [(x, y, bt) for x in range(WIDTH) for y in range(HEIGHT) for bt in range(5)]  # 가능한 모든 행동
        self.buildings = []  # 건물 저장

    def draw_grid(self):
        for c in range(0, WIDTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, HEIGHT * UNIT)
        for r in range(0, HEIGHT * UNIT, UNIT):
            self.canvas.create_line(0, r, WIDTH * UNIT, r)

    def is_building_at(self, x, y):
        # 해당 위치에 건물이 있는지 확인
        return any(bx == x and by == y for _, bx, by in self.buildings)

    def place_building(self, x, y, building_type):
        # 건물 배치
        if not self.is_building_at(x, y):
            self.buildings.append((building_type, x, y))
            self.update_canvas()  # 캔버스 업데이트
            return True
        return False

    def update_canvas(self):
        # 캔버스 업데이트
        self.canvas.delete("all")  # 캔버스 클리어
        self.draw_grid()  # 그리드 다시 그리기
        # 건물 그리기
        for building_type, x, y in self.buildings:
            color = ['purple', 'black', 'yellow', 'red', 'green'][building_type]  # 건물 유형별 색상
            self.canvas.create_rectangle(x * UNIT, y * UNIT, (x+1) * UNIT, (y+1) * UNIT, fill=color)

    def step(self, action_index):
        # 행동 실행
        action = self.actions[action_index]
        x, y, building_type = action
        placed = self.place_building(x, y, building_type)
        if placed:
            print(f"건물 {building_type}이(가) ({x},{y})에 배치되었습니다.")
        else:
            print(f"({x},{y})에는 이미 건물이 있습니다.")

if __name__ == "__main__":
    env = SmartCityEnvironment()
    env.after(1000, lambda: env.step(0))  # 테스트를 위한 첫 번째 행동 실행
    env.mainloop()
