import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class Node:
    def __init__(self, state, father=None, depth=0) -> None:
        self.state = state
        self.father = father
        self.depth = depth
        self.children = []
        self.f = 0

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def __str__(self) -> str:
        return (f"{self.state[1]} {self.state[2]} {self.state[3]}\n"
                f"{self.state[8]} {self.state[0]} {self.state[4]}\n"
                f"{self.state[7]} {self.state[6]} {self.state[5]}")

    def str2(self):
        return str(self.state)

    @classmethod
    def print(cls, container):
        item_strings = [item.str2() for item in container]
        print(f"{', '.join(item_strings)}")

    @classmethod
    def sort(cls, container) -> list:
        # 使用lambda函数作为排序关键字，根据实例的f属性进行排序
        sorted_container = sorted(container, key=lambda x: x.f)
        return sorted_container

    def calculate_f(self, target):
        w = 0
        for i, j in zip(self.state, target.state):
            if i != j:
                w += 1
        self.f = w + self.depth

    def extend(self):
        def swap(state, id1, id2) -> list:
            state_ = state[:]
            temp = state_[id2]
            state_[id2] = state_[id1]
            state_[id1] = temp
            return state_

        # 根据状态转换规则做扩展
        new_nodes = []
        if not self.state[0]:
            new_nodes.append(swap(self.state, 0, 2))
            new_nodes.append(swap(self.state, 0, 4))
            new_nodes.append(swap(self.state, 0, 6))
            new_nodes.append(swap(self.state, 0, 8))

        elif not self.state[1]:
            new_nodes.append(swap(self.state, 1, 2))
            new_nodes.append(swap(self.state, 1, 8))

        elif not self.state[2]:
            new_nodes.append(swap(self.state, 2, 1))
            new_nodes.append(swap(self.state, 2, 0))
            new_nodes.append(swap(self.state, 2, 3))

        elif not self.state[3]:
            new_nodes.append(swap(self.state, 3, 2))
            new_nodes.append(swap(self.state, 3, 4))

        elif not self.state[4]:
            new_nodes.append(swap(self.state, 4, 0))
            new_nodes.append(swap(self.state, 4, 3))
            new_nodes.append(swap(self.state, 4, 5))

        elif not self.state[5]:
            new_nodes.append(swap(self.state, 5, 4))
            new_nodes.append(swap(self.state, 5, 6))

        elif not self.state[6]:
            new_nodes.append(swap(self.state, 6, 0))
            new_nodes.append(swap(self.state, 6, 7))
            new_nodes.append(swap(self.state, 6, 5))

        elif not self.state[7]:
            new_nodes.append(swap(self.state, 7, 6))
            new_nodes.append(swap(self.state, 7, 8))

        elif not self.state[8]:
            new_nodes.append(swap(self.state, 8, 0))
            new_nodes.append(swap(self.state, 8, 1))
            new_nodes.append(swap(self.state, 8, 7))

        returns = []
        for n in new_nodes:
            returns.append(Node(n, self, self.depth + 1))

        return returns


class Experiment:
    def __init__(self, start, end, approach):
        self.approaches = {"DFS": self.dfs, "BFS": self.bfs, "A*": self.a_star}
        self.open_list = []
        self.closed_list = []
        self.start = Node(start)
        self.target = Node(end)
        self.focus = self.start
        self.approach = self.approaches[approach]
        self.extended_new = []
        self.path = []

    def normal_search(self, mode):
        """
        :param mode: 0:DFS, 1:BFS, 2:A*
        :return:
        """
        round_cnt = 0
        dfs = mode == 0
        bfs = mode == 1
        a_ = mode == 2

        # 1.把初始节点放入Open表中
        if a_:
            self.start.calculate_f(self.target)
        self.open_list.append(self.start)
        while self.open_list:  # 2.若Open表为空，搜索失败，退出
            round_cnt += 1
            if a_:
                self.open_list = Node.sort(self.open_list)
            print("轮次: ", round_cnt)
            print("Open表:")
            Node.print(self.open_list)
            print("Closed表:")
            Node.print(self.closed_list)

            # 3.取Open表前面第一个节点N，放进Closed表，冠以编号n
            self.focus = self.open_list[0]
            del self.open_list[0]
            print("考察节点:")
            print(self.focus)

            self.closed_list.append(self.focus)
            # 4.若目标节点等于N，搜索成功，结束
            if self.target == self.focus:
                print("SUCCEED")
                return self.focus
            # 5.若N不可扩展，则转步2
            # 6.扩展N，将其所有子节点配上指向N的返回指针，一次放入Open表的首部，转步2
            self.extended_new = self.focus.extend()
            print("扩展结果:")
            Node.print(self.extended_new)
            if bfs or dfs:
                self.extended_new = [x for x in self.extended_new if
                                     x != self.focus and x not in self.open_list and x not in self.closed_list]
                self.focus.children = self.extended_new
                print("扩展结果:")
                Node.print(self.extended_new)
                if not self.extended_new:
                    continue
                if bfs:  # BFS
                    self.open_list = self.open_list + self.extended_new
                elif dfs:  # DFS
                    self.open_list = self.extended_new + self.open_list

            elif a_:
                for j in self.extended_new:
                    j.calculate_f(self.target)
                    if j not in self.open_list and j not in self.closed_list:
                        self.focus.children.append(j)
                        self.open_list.append(j)
                    elif j in self.open_list:
                        for i in range(len(self.open_list)):
                            if self.open_list[i] == j and self.open_list[i].f > j.f:
                                for x in range(len(self.open_list[i].father.children)):
                                    if self.open_list[i].father.children[x] == j:
                                        del self.open_list[i].father.children[x]
                                self.open_list[i] = j
                                self.focus.children.append(j)
                    elif j in self.closed_list:
                        for i in range(len(self.closed_list)):
                            if self.closed_list[i] == j and self.closed_list[i].f > j.f:
                                for x in range(len(self.closed_list[i].father.children)):
                                    if self.closed_list[i].father.children[x] == j:
                                        del self.closed_list[i].father.children[x]
                                del self.closed_list[i]
                                self.open_list.append(j)
                                self.focus.children.append(j)
            print("=== === === === === ===")
        print("FAILED")
        return None

    def dfs(self):
        print("DFS")
        result = self.normal_search(0)
        return result

    def bfs(self):
        print("BFS")
        result = self.normal_search(1)
        return result

    def a_star(self):
        print("A*")
        result = self.normal_search(2)
        return result

    def execute(self):
        result = self.approach()
        if result is None:
            return
        else:
            print("PATH:")
            i = result
            while i.father is not None:
                self.path.append(i)
                i = i.father
            for i in range(len(self.path)):
                print(self.path[-i - 1])
                print("------")

    def draw(self):
        G = nx.DiGraph()
        node = self.start

        def traverse(n):
            G.add_node(str(n))
            for i in n.children:
                G.add_edge(str(n), str(i))
                traverse(i)

        traverse(node)
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=6, font_color="black", )
        for i in range(len(self.path)):
            nx.draw_networkx_edges(G, pos, edgelist=[(str(self.path[-i - 1].father), str(self.path[-i - 1]))],
                                   edge_color="red",
                                   width=2.0)  # 设置特定边的颜色
        # 绘制树结构
        plt.show()


class Chessboard:
    """
    可视化
    """

    def __init__(self, state):
        self.state = state
        self.state_old = state
        self.image = np.zeros((300, 300, 3), dtype="uint8")
        self.image = cv2.line(self.image, (100, 0), (100, 300), color=(255, 255, 255), thickness=3)
        self.image = cv2.line(self.image, (200, 0), (200, 300), color=(255, 255, 255), thickness=3)
        self.image = cv2.line(self.image, (0, 100), (300, 100), color=(255, 255, 255), thickness=3)
        self.image = cv2.line(self.image, (0, 200), (300, 200), color=(255, 255, 255), thickness=3)
        self.centers = [(150, 150), (50, 50), (150, 50), (250, 50), (250, 150), (250, 250), (150, 250), (50, 250),
                        (50, 150)]

        self.image_show_last = None
        pass

    def show_state(self):
        if self.image_show_last is not None:
            image_show = self.image_show_last.copy()
            action = [0, 0]
            for i in range(9):
                if self.state[i] != self.state_old[i]:
                    if self.state_old[i] == 0:
                        action[1] = i
                    else:
                        action[0] = i

            image_show = cv2.arrowedLine(image_show, self.centers[action[0]], self.centers[action[1]], (0, 255, 0))
            cv2.imshow("res", image_show)
            cv2.waitKey(1000)
        image_show = self.image.copy()
        for p, num in zip(self.centers, self.state):
            image_show = cv2.putText(image_show, str(num), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        self.image_show_last = image_show
        cv2.imshow("res", image_show)
        cv2.waitKey(1000)

    def update(self, state):
        self.state_old = self.state
        self.state = state
        pass


if __name__ == "__main__":
    exp = Experiment([0, 2, 8, 3, 4, 5, 6, 7, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8], "A*")
    exp.execute()
    exp.draw()
