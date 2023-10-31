class Node:
    def __init__(self, v, id):
        self.value = v
        self.id = id

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.value == other.value
        return False
    def cal(self):
        self.id = 888


if __name__ == "__main__":
    listA = [Node(1, 1), Node(2, 2), Node(3, 1)]
    N = Node(3, 3)
    for n in listA:
        n.cal()

    print(listA[2].value)
    print(listA[2].id)


    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout

    # 创建一个空的有向图
    G = nx.DiGraph()

    # 添加节点
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_node("D")

    # 添加边并为某个边指定颜色
    G.add_edge("A", "B", color="blue")  # 这里指定边A->B的颜色为蓝色
    G.add_edge("A", "C")
    G.add_edge("B", "D")

    # 以分层结构绘制树
    pos = graphviz_layout(G, prog="dot")

    # 获取边的颜色信息
    edge_colors = [G[u][v].get("color", "black") for u, v in G.edges()]

    # 绘制树结构，并设置边的颜色
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=10, font_color="black",
            edge_color=edge_colors)
    nx.draw_networkx_edges(G, pos, edgelist=[("A", "B")], edge_color="blue", width=2.0)  # 设置特定边的颜色
    plt.show()


