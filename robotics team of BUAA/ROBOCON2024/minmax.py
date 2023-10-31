import cv2
import numpy as np
import copy


class Silos:
    def __init__(self) -> None:
        self.image = None
        self.silo = [[] for _ in range(5)]

    def update(self, index, is_ai) -> None:
        self.silo[index].append(is_ai)

    def update_frame(self, result=None) -> None:
        image = np.zeros((300, 500, 3), dtype="uint8")
        if result == 1:
            image[:, :, 1] = 25
            image[:, :, 0] = 25
            image[:, :, 2] = 127
        elif result == 2:
            image[:, :, 1] = 25
            image[:, :, 2] = 25
            image[:, :, 0] = 127
        elif result == 3:
            image[:, :, 0] = 25
            image[:, :, 2] = 25
            image[:, :, 1] = 127
        image += 127
        image = cv2.rectangle(image, (400, 0), (500, 25), (0, 255, 0), -1)
        image = cv2.putText(image, "skip", (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)

        for i in range(1, 5):
            image = cv2.line(image, (100 * i, 0), (100 * i, 300), color=(255, 255, 255), thickness=3)
        for i in range(1, 3):
            image = cv2.line(image, (0, 100 * i), (500, 100 * i), color=(255, 255, 255), thickness=3)
        for s in range(5):
            for g in range(len(self.silo[s])):
                if self.silo[s][g]:
                    cv2.circle(image, (s * 100 + 50, 300 - g * 100 - 50), 30, (231, 107, 33), thickness=-1)
                else:
                    cv2.circle(image, (s * 100 + 50, 300 - g * 100 - 50), 30, (103, 90, 237), thickness=-1)
        self.image = image

    def show(self) -> None:
        self.update_frame()
        cv2.imshow("Silos", self.image)

    def game_end(self, who) -> None:
        self.update_frame(who)

        cv2.imshow("Silos", self.image)

    def alert(self):
        self.update_frame()
        self.image = cv2.rectangle(self.image, (0, 0), (100, 25), (0, 127, 127), -1)
        self.image = cv2.putText(self.image, "please", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                 thickness=2)
        cv2.imshow("Silos", self.image)


class Node(Silos):
    def __init__(self, silo=None) -> None:
        # 从列表构造节点
        super(Node, self).__init__()
        if silo is None:
            silo = [[] for _ in range(5)]
        self.silo = silo
        self.who = True  # 1或节点 AI，0与节点 人
        self.value = 0  # 估价
        self.children = []  # 子节点
        self.father = None
        self.choice = 0  # 下的位置

    @classmethod
    def min_max(cls, head, depth_limit) -> int:
        """
        极小极大搜索，然后返回决策
        :param depth_limit: 搜索深度限制
        :param head: 头节点,Node类型
        :return:
        """

        # 生成博弈树
        nodes_last_layer = [head]
        for i in range(depth_limit):
            for n in nodes_last_layer:
                n.extend()
            nodes_last_layer = [child for n in nodes_last_layer for child in n.children]
        # 遍历博弈树，估价
        head.value = cls.__traverse(head)
        # 选择
        # TODO: 选择更快大胜的方法，更好地处理无所谓的情况
        li = []
        for ch in head.children:
            if ch.value == head.value:
                li.append(ch.choice)

        if 5 in li and len(li) > 1:
            for i in li:
                if i != 5:
                    return i
        else:
            return li[0]

    @classmethod
    def __traverse(cls, head) -> int:
        """
        遍历一个博弈树，对每一个节点估价
        :param head:
        :return:
        """
        # TODO：alpha-beta剪枝
        values = []
        if head.children:
            for n in head.children:
                n.value = cls.__traverse(n)
                values.append(n.value)
            if head.who:  # 或节点, max
                return max(values)
            else:  # 与节点, min
                return min(values)
        else:
            head.value = cls.evaluate(head)
            return head.value

    @classmethod
    def evaluate(cls, node) -> int:
        """
        对末端节点估价函数

        （本状态就能大胜的局）
        若是AI的已大胜局，则 e = 999
        若是人的已大胜局，则 e = -999

        （下个状态能大胜的局）
        若是AI的必胜局，则 e = 99
        若是人的必胜局，则 e = -99

        若是胜负未定局，则 e = e1-e2
            e1: 棋局上 AI已经取得的桶数*10 + AI将要拿下的桶数*8 + AI的潜在机会数
            e2: 棋局上 人已经取得的桶数*10 + 人将要拿下的桶数*8 + 人的潜在机会数
        :return: value
        """
        # 先判断死局（本状态就能大胜的局）
        whup = Game.check_result(node.silo)
        if whup == 1:
            return -999
        elif whup == 2:
            return 999
        elif whup == 3:
            return 0

        # 如果不是死局，就计算各方已经拿下的桶量，将要拿下的桶量，机会量
        cnt0_ed = 0  # 人已经拿下的桶数量
        cnt1_ed = 0  # AI已经拿下的桶数量
        cnt0_will = 0  # 人将要拿下的桶数量
        cnt1_will = 0  # AI将要拿下的桶数量
        cnt0_chance = 0  # 人的机会量
        cnt1_chance = 0  # AI的机会量
        for s in node.silo:
            if len(s) == 3:
                if s[2] == s[0] or s[2] == s[1]:
                    if s[2] == 0:
                        cnt0_ed += 1
                    else:
                        cnt1_ed += 1
            elif len(s) == 2:
                if node.who and node.who in s:  # 轮到AI并且这个桶里已经有AI的球
                    cnt1_will += 1
                if (not node.who) and (not node.who) in s:  # 轮到人并且这个桶里已经有人的球
                    cnt0_will += 1
                if node.who and not (node.who in s):  # 轮到AI并且这个桶里没有AI的球
                    cnt0_chance += 3
                if (not node.who) and not ((not node.who) in s):  # 轮到人并且这个桶里没有人的球
                    cnt1_chance += 3
            elif len(s) == 1:
                if s[0]:
                    cnt1_chance += 1
                else:
                    cnt0_chance += 1

        e1 = 10 * cnt1_ed + 8 * cnt1_will + cnt1_chance
        e2 = 10 * cnt0_ed + 8 * cnt0_will + cnt0_chance

        return e1 - e2

    def extend(self) -> None:
        extendable = Game.check_result(self.silo) is None
        if extendable:
            for s in range(5):
                if len(self.silo[s]) <= 2:
                    child = Node(copy.deepcopy(self.silo))
                    child.silo[s].append(self.who)
                    child.father = self
                    child.who = not self.who
                    child.choice = s
                    self.children.append(child)
            # 如果轮到AI
            if self.who:
                child = Node(copy.deepcopy(self.silo))
                child.father = self
                child.who = not self.who
                child.choice = 5
                self.children.append(child)


class Game:
    def __init__(self, offensive=False, depth_limit=3, user=True):
        self.offensive = offensive  # AI先手
        self.depth_limit = depth_limit
        self.silos = Silos()
        self.result = None  # 没结果，人赢，机器赢，平局
        self.clicked = False
        self.round_cnt = 0
        self.user = user

    def ai_offense(self):
        head = Node(self.silos.silo)
        choice = Node.min_max(head, self.depth_limit)
        if choice != 5:
            self.silos.update(choice, True)
            self.silos.show()
        else:
            self.silos.alert()
        cv2.waitKey(1)
        return

    def ai2_offense(self):
        # AI2相当于一个反着来的AI1
        silos = []
        for s in self.silos.silo:
            li = []
            for x in s:
                li.append(not x)
            silos.append(li)

        head = Node(silos)
        choice = Node.min_max(head, self.depth_limit)
        if choice != 5:
            self.silos.update(choice, False)
            self.silos.show()
        else:
            self.silos.alert()
        cv2.waitKey(1)
        return

    def people_offense(self):
        def mouse_callback(event, x, y, __, ___):
            if event == cv2.EVENT_LBUTTONDOWN and not self.clicked:
                if x > 400 and y < 25:
                    self.clicked = True
                    return
                if len(self.silos.silo[int(x / 100)]) <= 2:
                    self.clicked = True
                    self.silos.update(int(x / 100), False)
                    self.silos.show()
                    cv2.waitKey(1)
                    self.clicked = True
                else:
                    return

        self.clicked = False

        cv2.setMouseCallback('Silos', mouse_callback)
        while not self.clicked:
            cv2.waitKey(1)

        if self.clicked:
            return

    @staticmethod
    def check_result(silo):
        cnt1 = 0
        cnt2 = 0
        for s in silo:
            if len(s) == 3:
                if s[2] == s[0] or s[2] == s[1]:
                    if s[2] == 0:
                        cnt1 += 1
                    else:
                        cnt2 += 1
        if cnt1 == 3:
            return 1
        elif cnt2 == 3:
            return 2

        cnt = 0
        for s in silo:
            if len(s) == 3:
                cnt += 1
        if cnt == 5:
            return 3
        return None

    def start(self):
        self.silos.show()
        while True:
            self.round_cnt += 1
            if self.offensive:
                self.ai_offense()
                self.result = self.check_result(self.silos.silo)
                if self.result:
                    break

                self.people_offense() if self.user else self.ai2_offense()

                self.result = self.check_result(self.silos.silo)
                if self.result:
                    break
            else:
                self.people_offense() if self.user else self.ai2_offense()
                self.result = self.check_result(self.silos.silo)
                if self.result:
                    break
                self.ai_offense()
                self.result = self.check_result(self.silos.silo)
                if self.result:
                    break

        if self.result == 1:
            print("红方赢")
        elif self.result == 2:
            print("蓝方赢")
        elif self.result == 3:
            print("平局")
        self.silos.game_end(self.result)
        cv2.waitKey(0)


if __name__ == "__main__":
    game = Game(False, 5, True)  # True/False代表先后手，depth_limit代表搜索深度, True/False代表自己玩还是AI对打
    game.start()
