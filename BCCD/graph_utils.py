"""
因果图工具模块

提供带置信度的因果图表示和可视化功能。
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端确保图形显示
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
from typing import List, Dict, Tuple, Optional, Any
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

# 设置中文字体以修复中文字符显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ConfidenceGraph(GeneralGraph):
    """
    带置信度的因果图类，扩展了 GeneralGraph。
    
    该类用于表示带有边置信度分数的因果图，提供图的属性查询、
    可视化等功能。
    
    Attributes:
        nodes (List[GraphNode]): 图节点列表
        variable_names (List[str]): 变量名称列表
        graph (nx.DiGraph): NetworkX 有向图表示
        edge_confidence (Dict[Tuple[int, int], float]): 边的置信度字典
    """
    
    def __init__(self, nodes: List[GraphNode], variable_names: List[str]):
        """
        初始化带置信度的因果图。
        
        Args:
            nodes: GraphNode 对象列表
            variable_names: 变量名称列表
        """
        super().__init__(nodes=nodes)
        self.variable_names = variable_names
        self.n_vars = len(nodes)
        
        # NetworkX 图表示
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.n_vars))
        
        # 边置信度存储
        self.edge_confidence: Dict[Tuple[int, int], float] = {}
    
    def add_directed_edge(self, i: int, j: int, confidence: float = 1.0) -> None:
        """
        添加有向边 i → j。
        
        Args:
            i: 起始节点索引
            j: 终止节点索引
            confidence: 边的置信度分数
        """
        self.graph.add_edge(i, j)
        self.edge_confidence[(i, j)] = confidence
    
    def set_edge_confidence(self, i: int, j: int, confidence: float) -> None:
        """
        设置边的置信度。
        
        Args:
            i: 起始节点索引
            j: 终止节点索引
            confidence: 置信度分数
        """
        if self.graph.has_edge(i, j):
            self.edge_confidence[(i, j)] = confidence
    
    def get_edge_confidence(self, i: int, j: int) -> float:
        """
        获取边的置信度。
        
        Args:
            i: 起始节点索引
            j: 终止节点索引
        
        Returns:
            float: 边的置信度，如果边不存在返回 0.0
        """
        return self.edge_confidence.get((i, j), 0.0)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        获取因果图的邻接矩阵。
        
        Returns:
            np.ndarray: 邻接矩阵，形状为 (n_vars, n_vars)
                       adj[i, j] = 1 表示存在边 i → j
        
        Examples:
            >>> adj_matrix = graph.get_adjacency_matrix()
            >>> print(adj_matrix)
        """
        adj_matrix = np.zeros((self.n_vars, self.n_vars))
        for i, j in self.graph.edges():
            adj_matrix[i, j] = 1
        return adj_matrix
    
    def get_confidence_matrix(self) -> np.ndarray:
        """
        获取边置信度矩阵。
        
        Returns:
            np.ndarray: 置信度矩阵，形状为 (n_vars, n_vars)
                       conf[i, j] 表示边 i → j 的贝叶斯置信度
        
        Examples:
            >>> conf_matrix = graph.get_confidence_matrix()
            >>> print(f"X0 → X1 的置信度: {conf_matrix[0, 1]:.3f}")
        """
        conf_matrix = np.zeros((self.n_vars, self.n_vars))
        for (i, j), confidence in self.edge_confidence.items():
            conf_matrix[i, j] = confidence
        return conf_matrix
    
    def get_causal_graph(self) -> GeneralGraph:
        """
        获取 causal-learn 的 GeneralGraph 对象。
        
        Returns:
            GeneralGraph: causal-learn 标准图结构，可用于其他 causal-learn 工具
        
        Examples:
            >>> causal_graph = graph.get_causal_graph()
            >>> # 可以使用 causal-learn 的其他工具进一步分析
        """
        return self
    
    def visualize(
        self,
        show_confidence: bool = True,
        confidence_threshold: float = 0.0,
        layout: str = 'spring',
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 3000,
        font_size: int = 12,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """
        可视化因果图，支持置信度显示。
        
        使用 networkx 和 matplotlib 进行可视化，支持多种布局算法和置信度显示。
        边的颜色和粗细表示置信度的高低。
        
        **边的颜色含义**:
        
        边的颜色由**归一化后的置信度**决定，反映该边在当前图中的相对置信度排名：
        
        - 归一化公式: ``norm_conf = (实际置信度 - 最小置信度) / (最大置信度 - 最小置信度)``
        - **深蓝色** (norm_conf=0.0): 当前图中置信度**最低**的边
        - **绿色** (norm_conf=0.5): 中等置信度的边  
        - **深红色** (norm_conf=1.0): 当前图中置信度**最高**的边
        
        **边旁边的数字含义**:
        
        显示的是该边的**贝叶斯置信度的绝对值**（通常范围 0-1）：
        
        - 表示该因果关系存在的后验概率
        - 例如: 0.95 表示有 95% 的概率该因果关系存在
        - 可以跨不同的图进行比较
        
        **颜色与数字的关系**:
        
        - **颜色**: 相对排名，用于在**同一张图内**快速识别哪些边最可靠
        - **数字**: 绝对度量，用于精确分析和**跨图比较**
        - 颜色会自动归一化以利用完整色彩范围，增强视觉对比度
        - 数字保留原始统计信息，便于科学评估
        
        **示例**:
        
        假设图中有3条边的置信度分别为 0.95, 0.90, 0.85:
        
        - 边A (0.95): 显示 **0.95**，颜色为**深红色** (最高)
        - 边B (0.90): 显示 **0.90**，颜色为**绿色** (中等)
        - 边C (0.85): 显示 **0.85**，颜色为**深蓝色** (最低)
        
        Args:
            show_confidence (bool, optional): 是否在边上显示置信度。默认 True。
            confidence_threshold (float, optional): 只显示置信度高于此阈值的边。默认 0.0。
            layout (str, optional): 图布局算法。默认 'spring'。
            
                **可选值**:
                
                - ``'spring'``: 弹簧布局（Fruchterman-Reingold）
                - ``'circular'``: 环形布局
                - ``'kamada_kawai'``: Kamada-Kawai 布局
                - ``'planar'``: 平面布局（如果可能）
                - ``'shell'``: 壳状布局
                - ``'spectral'``: 谱布局
                
            figsize (Tuple[int, int], optional): 图形大小 (width, height)。默认 (12, 8)。
            node_size (int, optional): 节点大小。默认 3000。
            font_size (int, optional): 字体大小。默认 12。
            save_path (str, optional): 保存图形的路径。如果为 None，则显示图形。
            title (str, optional): 自定义标题。如果为 None，则使用默认标题。默认 None。
        
        Examples:
            基本用法:
            
            .. code-block:: python
            
                graph.visualize(
                    show_confidence=True,
                    confidence_threshold=0.3,
                    save_path='causal_graph.png'
                )
            
            使用不同布局:
            
            .. code-block:: python
            
                graph.visualize(layout='circular')
                graph.visualize(layout='kamada_kawai')
            
            自定义标题:
            
            .. code-block:: python
            
                graph.visualize(title='因果图 (噪声=0.5)')
        
        Note:
            - 边的**粗细**也表示置信度：置信度越高，边越粗
            - 边的**颜色**从深蓝（低置信度）经绿色到深红（高置信度）渐变
            - 右侧**颜色条**显示颜色到置信度数值的映射关系
            - 需要安装 matplotlib 和 networkx 库
        """
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == 'planar':
            try:
                pos = nx.planar_layout(self.graph)
            except nx.NetworkXException:
                warnings.warn("图不是平面图，使用 spring 布局")
                pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(self.graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph)
        else:
            warnings.warn(f"未知布局 '{layout}'，使用默认 'spring' 布局")
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color='lightblue',
            node_size=node_size,
            alpha=0.9,
            edgecolors='darkblue',
            linewidths=2
        )
        
        # 绘制节点标签
        labels = {i: self.variable_names[i] for i in range(self.n_vars)}
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels,
            font_size=font_size,
            font_weight='bold'
        )
        
        # 筛选要显示的边
        edges_to_draw = []
        edge_confidences = []
        edge_labels = {}
        
        for i, j in self.graph.edges():
            confidence = self.edge_confidence.get((i, j), 0.0)
            if confidence >= confidence_threshold:
                edges_to_draw.append((i, j))
                edge_confidences.append(confidence)
                if show_confidence:
                    edge_labels[(i, j)] = f'{confidence:.2f}'
        
        if not edges_to_draw:
            warnings.warn("没有边满足置信度阈值，图形将只显示节点")
        else:
            # 归一化置信度用于颜色映射
            if len(edge_confidences) > 0:
                min_conf = min(edge_confidences)
                max_conf = max(edge_confidences)
                if max_conf > min_conf:
                    norm_confidences = [(c - min_conf) / (max_conf - min_conf) 
                                       for c in edge_confidences]
                else:
                    norm_confidences = [0.5] * len(edge_confidences)
            else:
                norm_confidences = []
            
            # 绘制边
            # 创建自定义颜色映射:深蓝→绿→深红
            # 定义关键颜色点（RGB格式）
            colors_custom = [
                (0.0, 0.0, 0.5),      # 深蓝色 (norm_conf=0.0)
                (0.0, 0.8, 0.0),      # 绿色 (norm_conf=0.5)
                (0.8, 0.0, 0.0)       # 深红色 (norm_conf=1.0)
            ]
            positions = [0.0, 0.5, 1.0]
            cmap = mcolors.LinearSegmentedColormap.from_list('deep_bgr', 
                                                             list(zip(positions, colors_custom)))
            
            for idx, (edge, conf, norm_conf) in enumerate(zip(edges_to_draw, 
                                                              edge_confidences, 
                                                              norm_confidences)):
                # 使用深色自定义映射
                edge_color = cmap(norm_conf)
                
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    [edge],
                    edge_color=[edge_color],
                    width=1 + 3 * conf,  # 边宽度随置信度变化
                    alpha=1,  # 提高不透明度使颜色更深
                    arrowsize=75,
                    arrowstyle='-|>',
                    connectionstyle='arc3,rad=0.1'
                )
            
            # 绘制边标签（置信度）
            if show_confidence and edge_labels:
                nx.draw_networkx_edge_labels(
                    self.graph,
                    pos,
                    edge_labels,
                    font_size=font_size - 2,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
        
        # 生成标题：使用自定义标题或默认标题
        if title is None:
            title = (f'因果图 (PAG)\n'
                    f'变量数: {self.n_vars}, 边数: {len(edges_to_draw)}, '
                    f'平均置信度: {np.mean(edge_confidences) if edge_confidences else 0:.3f}')
        
        plt.title(title, fontsize=font_size + 2, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # 添加颜色条
        if edges_to_draw and show_confidence:
            sm = cm.ScalarMappable(
                cmap=cmap,
                norm=mcolors.Normalize(
                    vmin=min(edge_confidences) if edge_confidences else 0,
                    vmax=max(edge_confidences) if edge_confidences else 1
                )
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
            cbar.set_label('边置信度', rotation=270, labelpad=20, fontsize=font_size)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")
        else:
            plt.show()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取图的摘要信息。
        
        Returns:
            Dict: 包含以下信息的字典
                - n_variables: 变量数量
                - n_edges: 边数量
                - average_confidence: 平均边置信度
                - edges: 边列表及其置信度
        
        Examples:
            >>> summary = graph.get_summary()
            >>> print(f"图中有 {summary['n_edges']} 条因果边")
            >>> print(f"平均置信度: {summary['average_confidence']:.3f}")
        """
        edges_list = []
        for i, j in self.graph.edges():
            confidence = self.edge_confidence.get((i, j), 0.0)
            edges_list.append({
                'from': self.variable_names[i],
                'to': self.variable_names[j],
                'confidence': confidence
            })
        
        # 按置信度排序
        edges_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        avg_confidence = np.mean(list(self.edge_confidence.values())) if self.edge_confidence else 0.0
        
        summary = {
            'n_variables': self.n_vars,
            'n_edges': self.graph.number_of_edges(),
            'average_confidence': avg_confidence,
            'edges': edges_list
        }
        
        return summary
    
    def print_summary(self) -> None:
        """
        打印图的摘要信息。
        
        Examples:
            >>> graph.print_summary()
        """
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("因果图摘要")
        print("=" * 70)
        print(f"变量数量: {summary['n_variables']}")
        print(f"因果边数量: {summary['n_edges']}")
        print(f"平均边置信度: {summary['average_confidence']:.3f}")
        print("\n因果边列表（按置信度排序）:")
        print("-" * 70)
        
        for edge in summary['edges']:
            print(f"  {edge['from']:>10} → {edge['to']:<10}  置信度: {edge['confidence']:.3f}")
        
        print("=" * 70)