import math
import cv2

EPS = 0.1
FPS_SHOW = 50  # 过程展示帧率，每秒多少轮
IF_SAVE_VIDEO = True  # 是否保存视频
FPS_SAVE = 25  # 视频保存帧率
TIME_PER_SHOW = int(1000 / FPS_SHOW)

MAP_IMAGE = 'map.png'  # 地图名称
MAP_WIDTH = 20  # 地图宽度
MAP_HEIGHT = 20  # 地图高度
HEURISTIC_FUNCTION = 'L1'  # 启发函数种类 L1，L2，LN (无穷范数)，DIJ (迪杰斯特拉，0)
# 起点和终点
START_POINT = (17, 0)
END_POINT = (0, 19)


class Node:
    def __init__(self, x, y, isaccessible):
        self.x = x
        self.y = y
        self.position = (x, y)
        self.accessible = isaccessible  # 是否可进入
        self.f = 0.0  # f值
        self.g = 0.0  # g值
        self.father_posi = (-1, -1)  # 指向父节点的指针


class Map:
    def __init__(self, width, height, mapimg):
        mapimg_gray = cv2.cvtColor(mapimg, cv2.COLOR_BGR2GRAY)
        self.image = cv2.resize(mapimg_gray, (500, 500))
        self.image_show = cv2.resize(mapimg, (500, 500))
        self.shape = (width, height)
        self.width = width
        self.height = height
        self.bin = []  # 按列存储的二值地图
        self.__image2bin()

    def __image2bin(self):
        """
        将图片地图转成栅格地图
        """
        for w in range(self.width):  # 列，x
            col = []
            for j in range(self.height):  # 行，y
                point_read = Node(w, j, isaccessible=(self.image[int((j + 0.5) * (500 / self.width))][int((w + 0.5) * (500 / self.height))] >= 127))
                col.append(point_read)
            self.bin.append(col)


class HeuristicSearch:
    def __init__(self, map_obj, start_point, end_point, heuristic_function):
        """
        地图对象，起点，终点，启发函数类型 支持L1，L2，LN，DIJ
        """
        self.__map = map_obj
        self.__start_point = start_point
        self.__end_point = end_point
        # 启发函数种类
        self.__heuristic_func = heuristic_function

        # 当前考察节点
        self.__node_focus = start_point

        # open表和closed表
        self.__open_list = []
        self.__closed_list = []

        # 解路径
        self.solution = []

        # 可视化
        self.show_img = None
        # 搜索轮数
        self.count = 0
        # 视频保存
        if IF_SAVE_VIDEO:
            self.out = cv2.VideoWriter(str(self.__heuristic_func) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS_SAVE, (500, 500))

    def __call__(self):
        # 1.把起始节点放进open表
        self.__open_list.append(self.__start_point)
        while True:
            self.count += 1
            # 2.如果OPEN表是个空表，则失败退出，无解
            if len(self.__open_list) == 0:
                print("NO SOLUTION !!!")
                return
            # 3.从OPEN表中选择一个f值最小的节点i。 结果有几个节点合格，当其中有一个为目标节点时，则选择此目标节点，否则就选择其中任一个节点作为节点i.
            # 4.1 把节点i 从OPEN表中移出。
            node_i_posi = self.__update()
            self.__node_focus = node_i_posi
            self.display_process()  # 可视化当前搜索情况

            # 4.2 把i放入CLOSED的扩展节点表中.
            self.__closed_list.append(node_i_posi)
            # 5.如果i是个目标节点, 则成功退出, 求得一个解.
            if node_i_posi == self.__end_point:
                print("SOLUTION FOUND !!!")
                self.__trace_back()
                self.display_result()
                return
            # 6.扩展节点i
            self.__extend()
            # 转向2

    def __evalution_function(self, point):
        """
        计算估价函数，返回g和h值
        """
        # 计算h
        h = self.__heuristic_function(point)
        # 计算g值
        if abs(point[0] - self.__node_focus[0]) + abs(point[1] - self.__node_focus[1]) == 1:
            g = self.__map.bin[self.__node_focus[0]][self.__node_focus[1]].g + 1.0
        else:
            g = self.__map.bin[self.__node_focus[0]][self.__node_focus[1]].g + math.sqrt(2)
        return g, h

    def __heuristic_function(self, point):
        """
        启发函数，根据启发函数代号计算对应函数值
        """
        if self.__heuristic_func == "L1":  # 1范数，曼哈顿距离
            return abs(self.__end_point[0] - point[0]) + abs(self.__end_point[1] - point[1])
        if self.__heuristic_func == "L2":  # 2范数，欧氏距离
            return math.sqrt((self.__end_point[0] - point[0]) ** 2 + (self.__end_point[1] - point[1]) ** 2)
        if self.__heuristic_func == "LN":  # 无穷范数
            return max(abs(self.__end_point[0] - point[0]), abs(self.__end_point[1] - point[1]))
        if self.__heuristic_func == "DIJ":  # 0，退化为迪杰斯特拉
            return 0.0

    def __extend(self):
        """
        扩展当前考察的节点
        """
        J = []  # 可扩展的所有节点J
        u, d, l_, r = False, False, False, False
        ua, da, la, ra = False, False, False, False
        # 上
        if 0 <= self.__node_focus[1] - 1 <= self.__map.height - 1:
            u = True
            if self.__map.bin[self.__node_focus[0]][self.__node_focus[1] - 1].accessible:
                ua = True
                J.append((self.__node_focus[0], self.__node_focus[1] - 1))
        # 下
        if 0 <= self.__node_focus[1] + 1 <= self.__map.height - 1:
            d = True
            if self.__map.bin[self.__node_focus[0]][self.__node_focus[1] + 1].accessible:
                da = True
                J.append((self.__node_focus[0], self.__node_focus[1] + 1))
        # 左
        if 0 <= self.__node_focus[0] - 1 <= self.__map.width - 1:
            l_ = True
            if self.__map.bin[self.__node_focus[0] - 1][self.__node_focus[1]].accessible:
                la = True
                J.append((self.__node_focus[0] - 1, self.__node_focus[1]))
        # 右
        if 0 <= self.__node_focus[0] + 1 <= self.__map.width - 1:
            r = True
            if self.__map.bin[self.__node_focus[0] + 1][self.__node_focus[1]].accessible:
                ra = True
                J.append((self.__node_focus[0] + 1, self.__node_focus[1]))

        # 左上
        if l_ and u and (la or ua):
            if self.__map.bin[self.__node_focus[0] - 1][self.__node_focus[1] - 1].accessible:
                J.append((self.__node_focus[0] - 1, self.__node_focus[1] - 1))

        # 右上
        if r and u and (ra or ua):
            if self.__map.bin[self.__node_focus[0] + 1][self.__node_focus[1] - 1].accessible:
                J.append((self.__node_focus[0] + 1, self.__node_focus[1] - 1))

        # 左下
        if l_ and d and (la or da):
            if self.__map.bin[self.__node_focus[0] - 1][self.__node_focus[1] + 1].accessible:
                J.append((self.__node_focus[0] - 1, self.__node_focus[1] + 1))

        # 右下
        if r and d and (ra or da):
            if self.__map.bin[self.__node_focus[0] + 1][self.__node_focus[1] + 1].accessible:
                J.append((self.__node_focus[0] + 1, self.__node_focus[1] + 1))

        # 对于每一个扩展出的节点
        for j in J:
            # 计算其估价函数
            g, h = self.__evalution_function(j)
            f = g + h
            # 如果j不在OPEN表和CLOSED表中,则用估价函数f把它添入OPEN表.从j加一指向父辈节点i的指针。
            if not (j in self.__open_list or j in self.__closed_list):
                self.__open_list.append(j)
                self.__map.bin[j[0]][j[1]].f = f
                self.__map.bin[j[0]][j[1]].g = g
                self.__map.bin[j[0]][j[1]].father_posi = self.__node_focus
            # 如果j已在OPEN表或CLOSED表：
            else:
                # 如果新值较小
                if self.__map.bin[j[0]][j[1]].f - f > EPS:
                    # 以新值取代旧值
                    self.__map.bin[j[0]][j[1]].f = f
                    self.__map.bin[j[0]][j[1]].g = g
                    # 更新父辈节点
                    self.__map.bin[j[0]][j[1]].father_posi = self.__node_focus
                    # 如果在colsed表，则移回open表
                    if j in self.__closed_list:
                        self.__closed_list.remove(j)
                        self.__open_list.append(j)

    def __update(self):
        """
        根据启发函数值对open表排序,返回启发函数值最小的那个，如果有多个最小值，且里面有目标节点，就直接返回目标节点，否则从最小值节点里随机返回一个
        并且从open表删除被返回的那个元素
        """
        # 根据估价函数升序排列open表
        self.__open_list = sorted(self.__open_list, key=lambda n: self.__map.bin[n[0]][n[1]].f, reverse=False)
        # 看看值最小的几个里面有没有终点
        min_f = self.__map.bin[self.__open_list[0][0]][self.__open_list[0][1]].f
        for i in self.__open_list:
            if abs(self.__map.bin[i[0]][i[1]].f - min_f) > EPS:
                break
            if i == self.__end_point:
                self.__open_list.remove(i)
                return i
        re = self.__open_list[0]
        del self.__open_list[0]
        return re

    def __trace_back(self):
        """
        回溯，拿到解路径
        """
        j = self.__end_point
        solution_inv = []  # 反转的解路径
        while j != self.__start_point:
            solution_inv.append(j)
            j = self.__map.bin[j[0]][j[1]].father_posi
        solution_inv.append(self.__start_point)
        self.solution = list(reversed(solution_inv))
        print(self.solution)

    def display_process(self):
        """
        过程可视化
        """
        self.show_img = self.__map.image_show.copy()
        # 蓝色圈标记open表元素
        for block in self.__open_list:
            cv2.circle(self.show_img, (int((block[0] + 0.5) * (500 / self.__map.width)),
                                       int((block[1] + 0.5) * (500 / self.__map.height))),
                       int((250 / min(self.__map.height, self.__map.width))),
                       (255, 0, 0), 5)
        # 绿色圈标注当前考察的节点
        cv2.circle(self.show_img, (int((self.__node_focus[0] + 0.5) * (500 / self.__map.width)),
                                   int((self.__node_focus[1] + 0.5) * (500 / self.__map.height))), int((250 / min(self.__map.height, self.__map.width))),
                   (0, 255, 0), 5)
        # 红色圈标注closed元素
        for block in self.__closed_list:
            cv2.circle(self.show_img, (int((block[0] + 0.5) * (500 / self.__map.width)),
                                       int((block[1] + 0.5) * (500 / self.__map.height))),
                       int((250 / min(self.__map.height, self.__map.width))),
                       (0, 0, 255), 5)
        # 青色折线标注当前的路径
        block = self.__node_focus
        while block != self.__start_point:
            cv2.line(self.show_img, (int((block[0] + 0.5) * (500 / self.__map.width)),
                                     int((block[1] + 0.5) * (500 / self.__map.height))),
                     (int((self.__map.bin[block[0]][block[1]].father_posi[0] + 0.5) * (500 / self.__map.width)),
                      int((self.__map.bin[block[0]][block[1]].father_posi[1] + 0.5) * (500 / self.__map.height))), (255, 255, 0), 5)
            block = self.__map.bin[block[0]][block[1]].father_posi

        cv2.imshow(str(self.__heuristic_func), self.show_img)
        # 写入视频流
        if IF_SAVE_VIDEO:
            self.out.write(self.show_img)

        cv2.waitKey(TIME_PER_SHOW)

    def display_result(self):
        """
        展示结果
        """
        cv2.destroyAllWindows()
        cv2.imshow("result", self.show_img)

        s1 = 0  # 横线，竖线
        s2 = 0  # 斜线

        for i in range(len(self.solution) - 1):
            if abs(self.solution[i][0] - self.solution[i + 1][0]) + abs(self.solution[i][1] - self.solution[i + 1][1]) == 1:
                s1 += 1
            else:
                s2 += 1

        print("length: ", s1 + s2 * 1.414)

        print("searched rounds: ", self.count)
        cv2.waitKey(0)


if __name__ == "__main__":
    map_0 = Map(MAP_WIDTH, MAP_HEIGHT, cv2.imread(MAP_IMAGE))
    search_0 = HeuristicSearch(map_0, START_POINT, END_POINT, HEURISTIC_FUNCTION)
    search_0()
