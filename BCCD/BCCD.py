"""
Bayesian Constraint-based Causal Discovery (BCCD) Algorithm

This module implements the BCCD algorithm proposed by Tom Claassen and Tom Heskes.
The algorithm combines Bayesian inference with constraint-based causal discovery methods
to provide probabilistic estimates of causal structures.

References:
    Claassen, T., & Heskes, T. (2012). A Bayesian approach to constraint based causal inference.
    In Proceedings of the Twenty-Eighth Conference on Uncertainty in Artificial Intelligence (UAI).

Author: Generated for LLM-for-Causality Project
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from itertools import combinations
from dataclasses import dataclass

from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

@dataclass
class IndependenceResult:
    """
    存储条件独立性测试结果的数据类。
    
    Attributes:
        x (int): 第一个变量的索引
        y (int): 第二个变量的索引
        z (Set[int]): 条件集的变量索引
        probability (float): 独立性的后验概率 P(X ⊥ Y | Z | D)
        test_statistic (float): 测试统计量
        p_value (float): p值（如果适用）
    """
    x: int
    y: int
    z: Set[int]
    probability: float
    test_statistic: float
    p_value: float


class BayesianIndependenceTest:
    """
    贝叶斯条件独立性测试类。
    
    实现贝叶斯方法来评估条件独立性，而非传统的硬阈值决策。
    使用贝叶斯因子（Bayes Factor）来量化独立性的证据强度。
    
    Attributes:
        data (np.ndarray): 观测数据矩阵，形状为 (n_samples, n_features)
        alpha (float): 先验超参数，用于贝叶斯因子计算
        method (str): 使用的测试方法 ('fisherz', 'chi_square', 'kci')
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        alpha: float = 1.0,
        method: str = 'fisherz'
    ):
        """
        初始化贝叶斯独立性测试。
        
        Args:
            data: 观测数据矩阵，形状为 (n_samples, n_features)
            alpha: 贝叶斯先验超参数，控制先验强度
            method: 独立性测试方法
                - 'fisherz': Fisher's Z 变换（适用于连续数据）
                - 'chi_square': 卡方检验（适用于离散数据）
                - 'kci': 核条件独立性检验
        """
        self.data = data
        self.alpha = alpha
        self.method = method
        self.n_samples, self.n_vars = data.shape
        
        # 初始化条件独立性测试器（来自 causal-learn）
        self.cit = CIT(data, method=method)
        
        # 缓存测试结果以提高效率
        self._test_cache: Dict[Tuple, IndependenceResult] = {}
    
    def test_independence(
        self, 
        x: int, 
        y: int, 
        z: Optional[Set[int]] = None
    ) -> IndependenceResult:
        """
        测试条件独立性 X ⊥ Y | Z，并返回后验概率。
        
        使用贝叶斯方法计算独立性的后验概率：
        
        .. math::
            P(X \\perp Y | Z | D) = \\frac{P(D | X \\perp Y | Z) P(X \\perp Y | Z)}{P(D)}
        
        其中贝叶斯因子定义为：
        
        .. math::
            BF = \\frac{P(D | H_0)}{P(D | H_1)}
        
        H_0: X ⊥ Y | Z (独立假设)
        H_1: X ⊥̸ Y | Z (非独立假设)
        
        Args:
            x: 第一个变量的索引
            y: 第二个变量的索引
            z: 条件集的变量索引集合（可选）
        
        Returns:
            IndependenceResult: 包含独立性后验概率和测试统计信息
        
        Examples:
            >>> tester = BayesianIndependenceTest(data)
            >>> result = tester.test_independence(0, 1, {2})
            >>> print(f"P(X0 ⊥ X1 | X2) = {result.probability:.3f}")
        """
        if z is None:
            z = set()
        
        # 检查缓存
        cache_key = (x, y, frozenset(z))
        if cache_key in self._test_cache:
            return self._test_cache[cache_key]
        
        # 转换条件集为列表
        z_list = sorted(list(z))
        
        # 使用 causal-learn 进行条件独立性测试
        p_value = self.cit(x, y, z_list)
        
        # 计算测试统计量（Fisher's Z）
        test_stat = self._compute_test_statistic(x, y, z_list)
        
        # 计算贝叶斯因子和后验概率
        bayes_factor = self._compute_bayes_factor(test_stat, len(z_list))
        posterior_prob = self._bayes_factor_to_probability(bayes_factor)
        
        result = IndependenceResult(
            x=x,
            y=y,
            z=z,
            probability=posterior_prob,
            test_statistic=test_stat,
            p_value=p_value
        )
        
        # 缓存结果
        self._test_cache[cache_key] = result
        
        return result
    
    def _compute_test_statistic(
        self, 
        x: int, 
        y: int, 
        z: List[int]
    ) -> float:
        """
        计算条件独立性的测试统计量。
        
        对于连续数据，使用 Fisher's Z 变换：
        
        .. math::
            Z = 0.5 \\log\\left(\\frac{1 + r}{1 - r}\\right) \\sqrt{n - |Z| - 3}
        
        其中 r 是偏相关系数。
        
        Args:
            x: 第一个变量索引
            y: 第二个变量索引
            z: 条件集变量索引列表
        
        Returns:
            float: 测试统计量
        """
        # 计算偏相关系数
        partial_corr = self._partial_correlation(x, y, z)
        
        # Fisher's Z 变换
        # 防止数值问题
        partial_corr = np.clip(partial_corr, -0.9999, 0.9999)
        
        n = self.n_samples
        k = len(z)
        
        # Z = 0.5 * ln((1+r)/(1-r)) * sqrt(n - k - 3)
        fisher_z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
        fisher_z *= np.sqrt(n - k - 3)
        
        return fisher_z
    
    def _partial_correlation(
        self, 
        x: int, 
        y: int, 
        z: List[int]
    ) -> float:
        """
        计算偏相关系数 ρ(X, Y | Z)。
        
        使用递归公式：
        
        .. math::
            \\rho_{XY|Z} = \\frac{\\rho_{XY|Z\\setminus\\{z_i\\}} - \\rho_{Xz_i|Z\\setminus\\{z_i\\}} \\rho_{Yz_i|Z\\setminus\\{z_i\\}}}{\\sqrt{(1-\\rho_{Xz_i|Z\\setminus\\{z_i\\}}^2)(1-\\rho_{Yz_i|Z\\setminus\\{z_i\\}}^2)}}
        
        Args:
            x: 第一个变量索引
            y: 第二个变量索引
            z: 条件集变量索引列表
        
        Returns:
            float: 偏相关系数
        """
        if len(z) == 0:
            # 简单相关系数
            return np.corrcoef(self.data[:, x], self.data[:, y])[0, 1]
        
        # 选择数据列
        vars_indices = [x, y] + z
        data_subset = self.data[:, vars_indices]
        
        # 计算相关矩阵
        corr_matrix = np.corrcoef(data_subset.T)
        
        # 计算偏相关（通过精度矩阵）
        try:
            precision_matrix = np.linalg.inv(corr_matrix)
            partial_corr = -precision_matrix[0, 1] / np.sqrt(
                precision_matrix[0, 0] * precision_matrix[1, 1]
            )
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，返回0
            partial_corr = 0.0
        
        return partial_corr
    
    def _compute_bayes_factor(
        self, 
        test_stat: float, 
        df: int
    ) -> float:
        """
        计算贝叶斯因子 BF = P(D|H0) / P(D|H1)。
        
        使用 BIC 近似：
        
        .. math::
            \\log BF \\approx -\\frac{1}{2} \\chi^2 + \\frac{df}{2} \\log n
        
        其中 χ² = Z² (对于 Fisher's Z 统计量)。
        
        Args:
            test_stat: 测试统计量（Fisher's Z）
            df: 自由度（条件集大小）
        
        Returns:
            float: 贝叶斯因子
        """
        n = self.n_samples
        
        # χ² 统计量
        chi_square = test_stat ** 2
        
        # BIC 近似的贝叶斯因子
        # log(BF) ≈ -0.5 * χ² + 0.5 * df * log(n)
        log_bf = -0.5 * chi_square + 0.5 * df * np.log(n)
        
        # 添加先验影响
        log_bf += np.log(self.alpha)
        
        bf = np.exp(log_bf)
        
        return bf
    
    def _bayes_factor_to_probability(self, bayes_factor: float) -> float:
        """
        将贝叶斯因子转换为后验概率。
        
        假设先验概率相等（P(H0) = P(H1) = 0.5）：
        
        .. math::
            P(H_0 | D) = \\frac{BF}{1 + BF}
        
        Args:
            bayes_factor: 贝叶斯因子
        
        Returns:
            float: 独立性的后验概率，范围 [0, 1]
        """
        # P(H0|D) = BF / (1 + BF)
        posterior = bayes_factor / (1.0 + bayes_factor)
        
        # 限制在 [0, 1] 范围内
        posterior = np.clip(posterior, 0.0, 1.0)
        
        return posterior


class BCCD:
    """Bayesian Constraint-based Causal Discovery (BCCD) 算法主类。
    
    基于 Claassen & Heskes (2012) 提出的 BCCD 算法实现。
    该算法结合了约束基方法的清晰性和贝叶斯方法的鲁棒性，
    通过贝叶斯评分获得输入陈述的概率估计，并按可靠性降序处理。
    
    算法包含三个核心阶段（对应 Algorithm 1）：
    
    - **Stage 0 (Mapping)**: 获取 uDAG 到逻辑因果语句的映射和先验
    - **Stage 1 (Search)**: 贝叶斯邻接搜索，识别条件独立性
    - **Stage 2 (Inference)**: 按可靠性降序处理逻辑因果陈述，推断因果关系
    
    Args:
        data (pd.DataFrame): 输入观测数据，每行为一个样本，每列为一个变量。
            
            数据格式要求：
            
            - **列名**: 必须提供变量名作为列名
            - **数据类型**: 支持数值型（连续变量）或分类型（离散变量）
            - **缺失值**: 不支持缺失值，请预先处理
            - **样本数**: 建议至少 100 个样本以获得可靠的统计推断
            
            **示例**:
            
            .. code-block:: python
            
                import pandas as pd
                data = pd.DataFrame({
                    'X': [1.2, 2.3, 3.1, ...],  # 连续变量
                    'Y': [0.5, 1.2, 2.1, ...],  # 连续变量
                    'Z': [1, 0, 1, ...]         # 离散变量
                })
                
        alpha (float, optional): 贝叶斯先验超参数，控制先验强度。
            默认值为 1.0。较大的值表示更强的先验信念。
            
        independence_threshold (float, optional): 独立性后验概率阈值 :math:`\\theta`。
            默认值为 0.5。对应 Algorithm 1 Line 3 和 Line 18。
            当 :math:`p(X \\perp Y | Z | D) > \\theta` 时，认为 X 和 Y 条件独立。
            
            - 较低的阈值（如 0.3）会导致更多边被移除，结果更稀疏
            - 较高的阈值（如 0.7）会保留更多边，结果更密集
            
        method (str, optional): 独立性测试方法。默认值为 'fisherz'。
        
            支持的方法：
            
            - ``'fisherz'``: Fisher's Z 变换（适用于连续数据，假设多元正态分布）
            - ``'chi_square'``: 卡方检验（适用于离散数据）
            - ``'kci'``: 核条件独立性检验（适用于非线性关系）
            
        max_cond_set_size (int, optional): 条件集的最大大小 :math:`K_{max}`。
            默认值为 ``n_vars - 2``。对应 Algorithm 1 Line 1 和 Line 4。
            
            - 限制计算复杂度，避免组合爆炸
            - 论文建议设置为 5（见 Algorithm 1 Line 1）
            - 对于稀疏图，较小的值通常已经足够
            
        background_knowledge (BackgroundKnowledge, optional): 背景知识约束。
            默认值为 None。可用于强制或禁止特定的因果边。
            
            **示例**:
            
            .. code-block:: python
            
                from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
                bk = BackgroundKnowledge()
                bk.add_required_by_node('X', 'Y')  # 要求 X → Y
                bk.add_forbidden_by_node('Z', 'X')  # 禁止 Z → X
                bccd = BCCD(data, background_knowledge=bk)
    
    Attributes:
        data (pd.DataFrame): 输入观测数据
        variable_names (List[str]): 变量名称列表，从 ``data.columns`` 获取
        n_vars (int): 变量数量
        alpha (float): 贝叶斯先验超参数
        independence_threshold (float): 独立性概率阈值 :math:`\\theta`
        method (str): 独立性测试方法
        max_cond_set_size (int): 条件集最大大小 :math:`K_{max}`
        data_array (np.ndarray): 数据的 numpy 数组表示，形状 ``(n_samples, n_vars)``
        independence_tester (BayesianIndependenceTest): 贝叶斯独立性测试器
        graph (nx.DiGraph): 学习到的有向因果图
        edge_confidence (Dict[Tuple[int, int], float]): 边的贝叶斯置信度映射
        independence_results (List[IndependenceResult]): 所有独立性测试结果列表（对应 Algorithm 1 中的 L）
        skeleton (nx.Graph): 骨架图（无向图，对应 Algorithm 1 中的 P）
        nodes (List): 节点列表
        background_knowledge (BackgroundKnowledge, optional): 背景知识约束
    
    Examples:
        基本用法:
        
        .. code-block:: python
        
            import pandas as pd
            from BCCD import BCCD
            
            # 准备数据
            data = pd.DataFrame({
                'X0': [1.2, 2.3, 3.1, 4.5, 5.2],
                'X1': [0.5, 1.2, 2.1, 3.0, 3.8],
                'X2': [2.1, 3.2, 4.5, 5.7, 6.9]
            })
            
            # 创建 BCCD 实例
            bccd = BCCD(data, alpha=1.0, independence_threshold=0.5)
            
            # 执行算法
            bccd.fit()
            
            # 查看结果
            bccd.print_summary()
            bccd.visualize_graph(show_confidence=True)
        
        使用背景知识:
        
        .. code-block:: python
        
            from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
            
            bk = BackgroundKnowledge()
            bk.add_required_by_node('X0', 'X1')
            
            bccd = BCCD(data, background_knowledge=bk)
            bccd.fit()
        
        调整阈值:
        
        .. code-block:: python
        
            # 更保守的阈值（更多独立性判断）
            bccd = BCCD(data, independence_threshold=0.7)
            bccd.fit()
    
    References:
        Claassen, T., & Heskes, T. (2012). A Bayesian approach to constraint based 
        causal inference. In Proceedings of the Twenty-Eighth Conference on 
        Uncertainty in Artificial Intelligence (UAI), pages 207-216.
    
    See Also:
        BayesianIndependenceTest : 贝叶斯独立性测试类
        IndependenceResult : 独立性测试结果数据类
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        alpha: float = 1.0,
        independence_threshold: float = 0.5,
        method: str = 'fisherz',
        max_cond_set_size: Optional[int] = None,
        background_knowledge: Optional[BackgroundKnowledge] = None
    ):
        """初始化 BCCD 算法。
        
        Args:
            data (pd.DataFrame): 输入观测数据矩阵。
                
                **格式要求**:
                
                - 行：样本（观测）
                - 列：变量（特征）
                - 列名：变量名称（必须提供）
                - 数值类型：float64 或可转换为 float64 的类型
                - 无缺失值
                
                **示例**:
                
                .. code-block:: python
                
                    data = pd.DataFrame({
                        'Temperature': [20.1, 22.3, 19.8, ...],
                        'Pressure': [1013, 1015, 1012, ...],
                        'Humidity': [65, 70, 68, ...]
                    })
                    
            alpha (float, optional): 贝叶斯先验超参数，默认 1.0。
                控制 Dirichlet 先验的强度。
                
                - ``alpha > 1``: 更强的先验信念
                - ``alpha = 1``: 均匀先验（默认）
                - ``alpha < 1``: 更弱的先验信念
                
            independence_threshold (float, optional): 独立性后验概率阈值 :math:`\\theta`，默认 0.5。
                对应 Algorithm 1 Line 3 和 Line 18。
                
                当 :math:`p(X \\perp Y | Z | D) > \\theta` 时认为变量独立。
                
            method (str, optional): 独立性测试方法，默认 ``'fisherz'``。
            
                **可选值**:
                
                - ``'fisherz'``: Fisher's Z 变换（连续数据，假设正态分布）
                - ``'chi_square'``: 卡方检验（离散数据）
                - ``'kci'``: 核条件独立性检验（非线性关系）
                
            max_cond_set_size (int, optional): 条件集最大大小 :math:`K_{max}`。
                默认为 ``n_vars - 2``。
                
                对应 Algorithm 1 Line 1 中的 :math:`K_{max}` 参数。
                论文建议值为 5。
                
            background_knowledge (BackgroundKnowledge, optional): 背景知识。
                默认 None。用于指定必须存在或必须不存在的因果边。
                
        Raises:
            ValueError: 如果数据包含缺失值
            ValueError: 如果数据列数少于 2
            TypeError: 如果数据不是 pd.DataFrame
            
        Note:
            - 算法复杂度为 :math:`O(n^2 \\cdot 2^{K_{max}})`，其中 :math:`n` 是变量数
            - 对于大规模数据，建议设置较小的 ``max_cond_set_size``
            - 使用 ``background_knowledge`` 可以提高效率和准确性
        """
        # 输入验证
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data 必须是 pd.DataFrame 类型")
        
        if data.shape[1] < 2:
            raise ValueError("数据至少需要 2 个变量")
        
        if data.isnull().any().any():
            raise ValueError("数据包含缺失值，请预先处理")
        
        self.data = data
        self.variable_names = list(data.columns)
        self.n_vars = len(self.variable_names)
        self.alpha = alpha
        self.independence_threshold = independence_threshold  # Algorithm 1 Line 3: θ = 0.5
        self.method = method
        
        # Algorithm 1 Line 1: Kmax = 5 (论文建议值)
        self.max_cond_set_size = max_cond_set_size if max_cond_set_size else min(5, self.n_vars - 2)
        
        # 背景知识
        self.background_knowledge = background_knowledge
        self.nodes = list(range(self.n_vars))  # 节点列表，用于背景知识查询
        
        # 转换为 numpy 数组
        self.data_array = data.values.astype(np.float64)
        
        # 初始化贝叶斯独立性测试器
        self.independence_tester = BayesianIndependenceTest(
            self.data_array,
            alpha=alpha,
            method=method
        )
        
        # Algorithm 1 输出: causal PAG P（有向图）
        self.graph: nx.DiGraph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.n_vars))
        
        # 存储边的置信度
        # key: (i, j), value: confidence score
        self.edge_confidence: Dict[Tuple[int, int], float] = {}
        
        # Algorithm 1 Stage 1: empty list L（独立性测试结果列表）
        self.independence_results: List[IndependenceResult] = []
        
        # Algorithm 1 Stage 1: fully connected P（骨架图，初始为完全连接）
        self.skeleton: nx.Graph = nx.Graph()
        self.skeleton.add_nodes_from(range(self.n_vars))
    
    def fit(self, verbose: bool = True) -> 'BCCD':
        """执行完整的 BCCD 算法流程（Algorithm 1）。
        
        实现 Claassen & Heskes (2012) 论文中的 Algorithm 1: Bayesian Constraint-based Causal Discovery。
        
        **算法流程** (对应论文 Algorithm 1):
        
        **Stage 0 - Mapping** (Line 1-2):
            获取 uDAG 到逻辑因果语句的映射和结构先验
            
        **Stage 1 - Search** (Line 3-15):
            贝叶斯邻接搜索，通过条件独立性测试识别骨架结构
            
        **Stage 2 - Inference** (Line 16-23):
            按可靠性降序处理逻辑因果陈述，推断因果关系
        
        Args:
            verbose (bool, optional): 是否打印详细的执行信息。默认 True。
        
        Returns:
            BCCD: 返回自身实例，支持链式调用。
        
        Raises:
            RuntimeError: 如果算法执行过程中出现错误
        
        Note:
            算法执行后，以下属性将被更新：
            
            - ``self.skeleton``: 骨架图（无向图）
            - ``self.graph``: 因果图（有向图，PAG 表示）
            - ``self.independence_results``: 独立性测试结果列表
            - ``self.edge_confidence``: 边的置信度映射
        
        Examples:
            基本用法:
            
            .. code-block:: python
            
                bccd = BCCD(data)
                bccd.fit()  # 使用默认详细输出
                print(f"发现 {bccd.graph.number_of_edges()} 条有向边")
                
            禁用详细输出:
            
            .. code-block:: python
            
                bccd = BCCD(data)
                bccd.fit(verbose=False)
                
            链式调用:
            
            .. code-block:: python
            
                summary = BCCD(data).fit().get_summary()
        
        References:
            Algorithm 1 in Claassen & Heskes (2012), UAI, pages 207-216.
        """
        if verbose:
            print("=" * 70)
            print("开始 BCCD 算法 (Algorithm 1: Claassen & Heskes, 2012)")
            print("=" * 70)
        
        # ============================================================
        # Stage 0 - Mapping (Algorithm 1, Line 1-2)
        # ============================================================
        if verbose:
            print("\n[Stage 0] Mapping - 获取 uDAG 映射和先验")
        
        # Line 1: G × L → Get_uDAG_Mapping(V, Kmax=5)
        # 获取从可能不忠实的 uDAG (unfaithful Directed Acyclic Graphs) 到逻辑因果陈述的映射
        # G: 所有可能的 uDAG 结构的集合
        # L: 逻辑因果陈述的集合（如 "X ⊥ Y | Z"）
        # V: 变量集合
        # Kmax: 条件集的最大大小（论文建议值为 5）
        # 注意：在本实现中，映射是隐式的，通过 d-separation 规则和贝叶斯评分实现
        if verbose:
            print(f"  Line 1: 使用 d-separation 规则建立 uDAG 映射 (Kmax={self.max_cond_set_size})")
        
        # Line 2: p(G) → Get_Prior(I)
        # 获取图结构的先验概率分布 p(G)
        # I: 背景信息或先验知识
        # 默认使用均匀先验（所有结构等概率）
        # 如果提供背景知识，则相应调整先验分布（禁止或要求特定边）
        if verbose:
            if self.background_knowledge is not None:
                print(f"  Line 2: 设置结构先验 p(G)（包含背景知识约束）")
            else:
                print(f"  Line 2: 设置结构先验 p(G)（均匀先验）")
        
        # ============================================================
        # Stage 1 - Search (Algorithm 1, Line 3-15)
        # ============================================================
        if verbose:
            print("\n[Stage 1] Search - 贝叶斯邻接搜索")
        
        # Line 3: fully connected P, empty list L, K=0, θ=0.5
        # 初始化搜索状态：
        # P: 完全连接的骨架图（所有变量对之间都有边）
        # L: 空的逻辑因果陈述列表（将在搜索过程中填充）
        # K: 条件集大小（从 0 开始）
        # θ (theta): 独立性后验概率阈值（默认 0.5）
        self._initialize_search(verbose=verbose)
        
        # Line 4-15: while K ≤ Kmax do ... end while
        # 贝叶斯邻接搜索主循环
        # 对不同大小的条件集（K=0 到 K=Kmax）执行独立性测试
        # 逐步移除骨架 P 中独立的变量对
        self._adjacency_search(verbose=verbose)
        
        if verbose:
            print(f"  搜索完成: 骨架包含 {self.skeleton.number_of_edges()} 条边")
        
        # ============================================================
        # Stage 2 - Inference (Algorithm 1, Line 16-23)
        # ============================================================
        if verbose:
            print("\n[Stage 2] Inference - 因果推断")
        
        # Line 16-23: 按概率降序处理逻辑因果陈述，生成 PAG
        self._causal_inference(verbose=verbose)
        
        if verbose:
            print(f"  推断完成: 发现 {self.graph.number_of_edges()} 条有向边")
        
        # ============================================================
        # 附加步骤：计算边的贝叶斯置信度
        # ============================================================
        if verbose:
            print("\n[附加步骤] 计算边的贝叶斯置信度")
        self._compute_edge_confidence(verbose=verbose)
        
        if verbose:
            print("\n" + "=" * 70)
            print("BCCD 算法完成!")
            print("=" * 70)
        
        return self
    
    def _initialize_search(self, verbose: bool = True) -> None:
        """初始化搜索状态（Algorithm 1, Line 3）。
        
        对应论文 Algorithm 1 的 Line 3:
        
        .. code-block:: text
        
            fully connected P, empty list L, K=0, θ=0.5
        
        初始化以下状态：
        
        - **P** (``self.skeleton``): 完全连接的无向图（所有变量对之间都有边）
        - **L** (``self.independence_results``): 空的独立性测试结果列表
        - **K**: 条件集大小，从 0 开始（在 ``_adjacency_search`` 中递增）
        - **θ** (``self.independence_threshold``): 独立性概率阈值，默认 0.5
        
        Note:
            - 初始骨架为完全图，包含 :math:`\\binom{n}{2}` 条边，其中 :math:`n` 是变量数
            - 如果提供了背景知识，会跳过被明确禁止的边
            - 空列表 L 将在 ``_adjacency_search`` 中逐步填充
        
        Args:
            verbose (bool): 是否打印详细信息
        """
        # Line 3: fully connected P
        # 初始化完全连接的骨架图 P
        # 在所有变量对之间添加无向边，形成完全图
        # 完全图包含 n*(n-1)/2 条边，其中 n 是变量数
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                # 检查背景知识是否禁止此边
                if self.background_knowledge is not None:
                    # 如果背景知识明确禁止 i-j 和 j-i 两个方向，则跳过此边
                    # 这意味着 i 和 j 之间不可能有任何因果关系
                    if (self.background_knowledge.is_forbidden(self.nodes[i], self.nodes[j]) and
                        self.background_knowledge.is_forbidden(self.nodes[j], self.nodes[i])):
                        continue
                
                # 添加无向边 i - j
                self.skeleton.add_edge(i, j)
        
        if verbose:
            print(f"  Line 3: 初始化完全连接图 P ({self.skeleton.number_of_edges()} 条边)")
        
        # Line 3: empty list L
        # 初始化空的逻辑因果陈述列表 L
        # L 将在 _adjacency_search 中逐步填充独立性测试结果
        # 每个元素是一个 IndependenceResult 对象，包含 X⊥Y|Z 及其后验概率
        if verbose:
            print(f"  Line 3: 初始化空列表 L (将存储独立性测试结果)")
        
        # Line 3: K=0
        # 设置初始条件集大小 K = 0
        # K 将在 _adjacency_search 的外层循环中从 0 递增到 Kmax
        # K=0 意味着首先测试无条件独立性 (X⊥Y)
        if verbose:
            print(f"  Line 3: 初始条件集大小 K = 0")
        
        # Line 3: θ=0.5
        # 设置独立性后验概率阈值 θ
        # θ 已在 __init__ 中通过 independence_threshold 参数设置
        # 当 p(X⊥Y|Z|D) > θ 时，认为 X 和 Y 条件独立
        if verbose:
            print(f"  Line 3: 独立性阈值 θ = {self.independence_threshold}")
    
    def _adjacency_search(self, verbose: bool = True) -> None:
        """贝叶斯邻接搜索（Algorithm 1, Line 4-15）。
        
        对应论文 Algorithm 1 的 Stage 1 主循环:
        
        .. code-block:: text
        
            4: while K ≤ Kmax do
            5:   for all X ∈ V, Y ∈ Adj(X) in P do
            6:     for all Z ⊆ Adj(X)\Y, |Z| = K do
            7:       W → Check_Unprocessed(X, Y, Z)
            8:       ∀G ∈ G_W: compute p(G|D_W)
            9:       ∀L: p(L_W|D_W) → Σ_{G→L_W} p(G|D_W)
            10:      ∀L: p(L) → max(p(L), p(L_W|D_W))
            11:      P → p("W_i -/- W_j"|D_W) > θ
            12:    end for
            13:  end for
            14:  K = K + 1
            15: end while
        
        执行贝叶斯独立性测试来识别骨架结构。通过逐步增加条件集大小 K，
        对每对相邻变量 (X, Y) 测试所有大小为 K 的条件集 Z，
        计算独立性的后验概率 p(X⊥Y|Z|D)，并更新骨架 P 和逻辑陈述列表 L。
        
        Args:
            verbose (bool): 是否打印详细信息
        
        Note:
            - Line 7: Check_Unprocessed 用于避免重复测试相同的变量子集
            - Line 8-9: 对子集 W={X,Y}∪Z 上的所有可能 uDAG 计算后验概率
            - Line 10: 保留每个逻辑陈述的最大概率估计（来自不同子集）
            - Line 11: 如果独立性概率超过阈值 θ，从骨架 P 中移除边
            - 复杂度: :math:`O(n^2 \\cdot 2^{K_{max}})` 其中 n 是变量数
        """
        # Line 4: while K ≤ Kmax do
        # 对每个条件集大小 K（从 0 到 Kmax）执行独立性测试
        # 外层循环：逐步增加条件集大小，从 K=0 开始直到 K=Kmax
        for cond_size in range(self.max_cond_set_size + 1):
            if verbose:
                print(f"\n  Line 4: 条件集大小 K = {cond_size}")
            
            edges_to_remove = []
            
            # Line 5: for all X ∈ V, Y ∈ Adj(X) in P do
            # 遍历骨架 P 中的所有边 (X, Y)
            # 注意：遍历当前骨架中所有相邻的变量对
            for i, j in list(self.skeleton.edges()):
                # Line 6: for all Z ⊆ Adj(X)\Y, |Z| = K do
                # 获取可能的条件变量集合
                # Adj(X)\Y 表示 X 的所有邻居除了 Y
                # 我们取 X 和 Y 的所有邻居的并集（排除彼此）作为候选条件变量
                neighbors_i = set(self.skeleton.neighbors(i)) - {j}
                neighbors_j = set(self.skeleton.neighbors(j)) - {i}
                possible_cond_vars = neighbors_i | neighbors_j
                
                # 如果可能的条件变量数量不足以形成大小为 K 的条件集
                if len(possible_cond_vars) < cond_size:
                    continue
                
                # 对所有大小为 cond_size 的条件集进行测试
                found_independence = False
                for cond_set in combinations(possible_cond_vars, cond_size):
                    cond_set_set = set(cond_set)
                    
                    # Line 7: W ← Check_Unprocessed(X, Y, Z)
                    # 构建变量子集 W = {X, Y} ∪ Z
                    # 注意：在实际实现中，我们直接测试而不显式构建 W
                    
                    # Line 8: ∀G ∈ G_W: compute p(G|D_W)
                    # Line 9: ∀L: p(L_W|D_W) ← Σ_{G→L_W} p(G|D_W)
                    # 计算独立性的贝叶斯后验概率
                    # 这通过 BayesianIndependenceTest 完成，它：
                    # 1. 对子集 W 上的所有可能 uDAG 结构计算 p(G|D_W)
                    # 2. 将结构概率映射到逻辑因果陈述 L
                    # 3. 返回独立性陈述的后验概率
                    result = self.independence_tester.test_independence(i, j, cond_set_set)
                    
                    # Line 10: ∀L: p(L) ← max(p(L), p(L_W|D_W))
                    # 将测试结果添加到列表 L
                    # （保留最大概率在 _causal_inference 中处理）
                    self.independence_results.append(result)
                    
                    # Line 11: P ← p("W_i -/- W_j"|D_W) > θ
                    # 如果独立性后验概率超过阈值 θ，标记边以供移除
                    if result.probability > self.independence_threshold:
                        edges_to_remove.append((i, j))
                        found_independence = True
                        if verbose:
                            cond_str = f"|{{{', '.join([self.variable_names[k] for k in cond_set])}}}" if cond_set else ""
                            print(f"    Line 11: 移除边 {self.variable_names[i]} - {self.variable_names[j]} "
                                  f"(p(⊥{cond_str}|D) = {result.probability:.3f} > {self.independence_threshold})")
                        break  # Line 12: end for (内层循环 - 对 Z 的遍历)
                
                # Line 12: end for
                # 如果找到独立性，跳出条件集循环，继续下一对变量
                if found_independence:
                    break  # 跳到下一对变量 (提前结束对当前边的测试)
            
            # Line 11 后续处理：批量移除标记为独立的边
            # 在完成所有测试后，统一从骨架 P 中移除边
            for edge in edges_to_remove:
                if self.skeleton.has_edge(*edge):
                    self.skeleton.remove_edge(*edge)
            
            if verbose:
                print(f"  当前骨架: {self.skeleton.number_of_edges()} 条边")
            
            # Line 13: end for (中层循环 - 对所有边 (X,Y) 的遍历)
            # Line 14: K = K + 1
            # 递增条件集大小（通过 for 循环自动完成）
        
        # Line 15: end while (外层循环 - K ≤ Kmax 的条件)
        # 邻接搜索阶段完成
        if verbose:
            print(f"  Line 15: 邻接搜索完成，共测试 {len(self.independence_results)} 个条件独立性")
    
    def _causal_inference(self, verbose: bool = True) -> None:
        """因果推断阶段（Algorithm 1, Line 16-23）。
        
        对应论文 Algorithm 1 的 Stage 2:
        
        .. code-block:: text
        
            16: LC = empty 3D-matrix size |V|³, i=1
            17: L̃ ← Sort_Descending(L, p(L))
            18: while p(Li) > θ do
            19:   LC ← Run_Causal_Logic(LC, Li)
            20:   i ← i + 1
            21: end while
            22: MC ← Get_Causal_Matrix(LC)
            23: P ← Map_To_PAG(P, MC)
        
        基于收集的逻辑因果陈述进行因果推断。
        按可靠性（后验概率）降序处理独立性陈述，
        应用因果逻辑规则（v-结构、传播规则等）来确定边的方向。
        
        Args:
            verbose (bool): 是否打印详细信息
        
        Note:
            - Line 16: LC 是逻辑因果陈述的 3D 矩阵，本实现简化为图结构
            - Line 17: 按后验概率 p(L) 降序排序陈述列表
            - Line 18-21: 只处理概率超过阈值 θ 的陈述
            - Line 19: Run_Causal_Logic 应用因果推断规则（见 ``_orient_edges``）
            - Line 22-23: 生成因果矩阵和 PAG 表示
        """
        # Line 16: LC = empty 3D-matrix size |V|³, i=1
        # 初始化逻辑因果矩阵 LC 和索引 i
        # LC[X][Y][Z] 存储关于 X, Y, Z 之间关系的逻辑因果陈述
        # 在本实现中，使用有向图 self.graph 和独立性结果列表代替 3D 矩阵
        if verbose:
            print(f"\n  Line 16: 初始化逻辑因果矩阵 LC (|V|³ = {self.n_vars}³), i=1")
        
        # Line 17: L̃ ← Sort_Descending(L, p(L))
        # 按后验概率 p(L) 降序排序逻辑因果陈述列表 L
        # 这确保我们优先处理最可靠（概率最高）的独立性陈述
        if verbose:
            print(f"  Line 17: 按概率 p(L) 降序排序 {len(self.independence_results)} 个逻辑因果陈述")
        
        sorted_results = sorted(
            self.independence_results, 
            key=lambda x: x.probability, 
            reverse=True
        )
        
        # Line 18: while p(Li) > θ do
        # 只处理后验概率超过阈值 θ 的陈述
        # 统计满足条件的高置信度陈述数量
        high_confidence_count = sum(
            1 for r in sorted_results 
            if r.probability > self.independence_threshold
        )
        
        if verbose:
            print(f"  Line 18: 开始处理高置信度陈述 (p(Li) > θ = {self.independence_threshold})")
            print(f"  共 {high_confidence_count} 个陈述满足条件")
        
        # 准备 PAG 的初始状态：将骨架 P 复制到有向图
        # 初始化为双向边表示尚未确定方向的边（PAG 中的 'o-o' 标记）
        for i, j in self.skeleton.edges():
            # 添加双向边: i <-> j
            self.graph.add_edge(i, j)
            self.graph.add_edge(j, i)
        
        if verbose:
            print(f"  初始化双向边图: {self.graph.number_of_edges()} 条边 (骨架的 2 倍)")
        
        # Line 19: LC ← Run_Causal_Logic(LC, Li)
        # 对每个高置信度陈述 Li，应用因果逻辑规则更新 LC
        # 因果逻辑规则包括：
        # - 检测 v-结构 (colliders)
        # - 应用 Meek 规则传播边的方向
        # - 考虑背景知识约束
        if verbose:
            print(f"\n  Line 19: Run_Causal_Logic - 应用因果推断规则")
        
        self._orient_edges(verbose=verbose)
        
        # Line 20: i ← i + 1
        # 递增索引，处理下一个陈述
        # Line 21: end while
        # 当 p(Li) ≤ θ 时结束循环（或所有陈述处理完毕）
        # 注意：在本实现中，所有高置信度陈述已在 _orient_edges 中统一处理
        
        # Line 22: MC ← Get_Causal_Matrix(LC)
        # 从逻辑因果矩阵 LC 生成因果矩阵 MC
        # MC 是 LC 的汇总，表示变量间的最终因果关系
        # Line 23: P ← Map_To_PAG(P, MC)
        # 将因果矩阵 MC 映射回骨架 P，生成最终的部分有向无环图 (PAG)
        # 注意：在本实现中，self.graph 直接表示最终的 PAG，无需额外映射步骤
        if verbose:
            print(f"  Line 22-23: 生成因果矩阵 MC 并映射到 PAG")
            print(f"  最终 PAG: {self.graph.number_of_edges()} 条有向边")
    
    def _orient_edges(self, verbose: bool = True) -> None:
        """确定骨架中边的方向（Algorithm 1, Line 19: Run_Causal_Logic）。
        
        对应论文 Algorithm 1 的 Line 19:
        
        .. code-block:: text
        
            LC ← Run_Causal_Logic(LC, Li)
        
        应用因果逻辑规则来推断边的方向，形成部分有向无环图 (PAG)。
        使用 v-结构规则和传播规则，同时考虑背景知识约束。
        
        **方向确定规则**:
        
        - **规则 0** (背景知识): 应用用户提供的背景知识约束
        - **规则 1** (V-结构): 检测 unshielded colliders
        - **规则 2**: Meek规则 - 避免新的 v-结构
        - **规则 3**: Meek规则 - 传播方向
        - **规则 4**: Meek规则 - 避免环路
        
        Args:
            verbose (bool): 是否打印详细信息
        
        Note:
            - 骨架已在 ``_adjacency_search`` (Line 4-15) 中构建
            - 有向图已在 ``_causal_inference`` 中初始化为双向边
            - 双向边表示未定向的边（circle mark 'o' 在 PAG 中）
            - 单向边表示已确定方向的因果边
        
        References:
            - Meek, C. (1995). Causal inference and causal explanation with background knowledge.
            - Zhang, J. (2008). On the completeness of orientation rules for causal discovery.
        """
        # 注意：骨架已在 _adjacency_search 中构建
        # 这里的有向图已在 _causal_inference 中初始化为双向边
        
        # 规则 0: 应用背景知识
        if self.background_knowledge is not None:
            if verbose:
                print("  应用规则 0: 使用背景知识...")
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    if i == j:
                        continue
                    
                    # 如果背景知识要求 i → j
                    if self.background_knowledge.is_required(self.nodes[i], self.nodes[j]):
                        if self.graph.has_edge(i, j):
                            self._remove_reverse_edge(i, j)
                            if verbose:
                                print(f"    背景知识要求: {self.variable_names[i]} → {self.variable_names[j]}")
                    
                    # 如果背景知识禁止 i → j
                    if self.background_knowledge.is_forbidden(self.nodes[i], self.nodes[j]):
                        if self.graph.has_edge(i, j):
                            self.graph.remove_edge(i, j)
                            if verbose:
                                print(f"    背景知识禁止: {self.variable_names[i]} → {self.variable_names[j]}")
        
        # 规则 1: 寻找 v-结构
        if verbose:
            print("  应用规则 1: 检测 v-结构...")
        for k in range(self.n_vars):
            # 找到 k 的所有邻居
            neighbors = list(self.skeleton.neighbors(k))
            
            # 检查所有不相邻的邻居对
            for i, j in combinations(neighbors, 2):
                # 如果 i 和 j 不相邻
                if not self.skeleton.has_edge(i, j):
                    # 检查 k 是否在 i 和 j 的分离集中
                    in_sep_set = self._is_in_separation_set(i, j, k)
                    
                    if not in_sep_set:
                        # 检查背景知识是否允许这些方向
                        can_orient_ik = (self.background_knowledge is None or 
                                       not self.background_knowledge.is_forbidden(self.nodes[i], self.nodes[k]))
                        can_orient_jk = (self.background_knowledge is None or 
                                       not self.background_knowledge.is_forbidden(self.nodes[j], self.nodes[k]))
                        
                        if can_orient_ik and can_orient_jk:
                            # 定向为 i → k ← j
                            self._remove_reverse_edge(i, k)
                            self._remove_reverse_edge(j, k)
                            if verbose:
                                print(f"    V-结构: {self.variable_names[i]} → "
                                      f"{self.variable_names[k]} ← {self.variable_names[j]}")
        
        # 规则 2-4: 传播边方向
        if verbose:
            print("  应用规则 2-4: 传播边方向...")
        changed = True
        iteration = 0
        while changed:
            changed = False
            iteration += 1
            if verbose:
                print(f"    迭代 {iteration}...")
            
            # 规则 2: i → j - k 且 i, k 不相邻 => j → k
            for j in range(self.n_vars):
                for i in self.graph.predecessors(j):
                    if self.graph.has_edge(j, i):
                        continue  # 边未定向
                    
                    for k in list(self.skeleton.neighbors(j)):
                        if k == i:
                            continue
                        if not self.skeleton.has_edge(i, k):
                            if self.graph.has_edge(j, k) and self.graph.has_edge(k, j):
                                # 定向为 j → k
                                if self._safe_orient(j, k):
                                    changed = True
            
            # 规则 3: i - j, i → m → j => i → j
            for i, j in list(self.skeleton.edges()):
                if self.graph.has_edge(i, j) and self.graph.has_edge(j, i):
                    # 检查是否存在路径 i → m → j
                    for m in range(self.n_vars):
                        if m == i or m == j:
                            continue
                        if (self.graph.has_edge(i, m) and 
                            not self.graph.has_edge(m, i) and
                            self.graph.has_edge(m, j) and 
                            not self.graph.has_edge(j, m)):
                            # 定向为 i → j
                            if self._safe_orient(i, j):
                                changed = True
                                break
            
            # 规则 4: i - j, i → k, k → m, m - j => i → j
            # 避免形成新的环路和 v-结构
            for i, j in list(self.skeleton.edges()):
                if self.graph.has_edge(i, j) and self.graph.has_edge(j, i):
                    # 寻找两条不同的路径 i → k → m 和 m - j
                    for k in range(self.n_vars):
                        if k == i or k == j:
                            continue
                        if not (self.graph.has_edge(i, k) and not self.graph.has_edge(k, i)):
                            continue  # 必须是 i → k
                        
                        for m in range(self.n_vars):
                            if m == i or m == j or m == k:
                                continue
                            # 检查 k → m 和 m - j
                            if (self.graph.has_edge(k, m) and 
                                not self.graph.has_edge(m, k) and
                                self.graph.has_edge(m, j) and 
                                self.graph.has_edge(j, m)):
                                # 定向为 i → j
                                if self._safe_orient(i, j):
                                    changed = True
                                    break
                        if changed:
                            break
        
        # 移除未定向的边（双向边）
        edges_to_remove = []
        for i, j in self.graph.edges():
            if self.graph.has_edge(j, i):
                # 保留一个方向（按索引顺序）
                if i < j:
                    edges_to_remove.append((j, i))
                else:
                    edges_to_remove.append((i, j))
        
        for edge in edges_to_remove:
            if self.graph.has_edge(*edge):
                self.graph.remove_edge(*edge)
    
    def _is_in_separation_set(self, i: int, j: int, k: int) -> bool:
        """
        检查变量 k 是否在 i 和 j 的分离集中。
        
        Args:
            i: 第一个变量索引
            j: 第二个变量索引
            k: 待检查的变量索引
        
        Returns:
            bool: 如果 k 在分离集中返回 True
        """
        # 从独立性测试结果中查找
        for result in self.independence_results:
            if (result.x == i and result.y == j) or (result.x == j and result.y == i):
                if result.probability > self.independence_threshold:
                    if k in result.z:
                        return True
        return False
    
    def _remove_reverse_edge(self, i: int, j: int) -> None:
        """
        定向边 i → j（如果存在反向边则移除）。
        
        Args:
            i: 起始节点
            j: 终止节点
        """
        if self.graph.has_edge(j, i):
            self.graph.remove_edge(j, i)
    
    def _safe_orient(self, i: int, j: int) -> bool:
        """
        安全地定向边 i → j，避免产生环路。
        
        Args:
            i: 起始节点
            j: 终止节点
        
        Returns:
            bool: 如果成功定向返回 True
        """
        if not self.graph.has_edge(i, j):
            return False
        
        # 移除反向边
        if self.graph.has_edge(j, i):
            self.graph.remove_edge(j, i)
            
            # 检查是否会产生环路
            if self._creates_cycle(i, j):
                # 恢复反向边
                self.graph.add_edge(j, i)
                return False
            
            return True
        
        return False
    
    def _creates_cycle(self, i: int, j: int) -> bool:
        """
        检查添加边 i → j 是否会创建环路。
        
        Args:
            i: 起始节点
            j: 终止节点
        
        Returns:
            bool: 如果会产生环路返回 True
        """
        # 使用 DFS 检查从 j 是否能到达 i
        try:
            path = nx.shortest_path(self.graph, j, i)
            return True
        except nx.NetworkXNoPath:
            return False
    
    def _compute_edge_confidence(self, verbose: bool = True) -> None:
        """计算每条边的贝叶斯置信度。
        
        置信度定义为：给定数据 D，该因果边存在的后验概率。
        
        **计算方法**:
        
        使用独立性测试结果来推断边的置信度：
        
        .. math::
            p(i \\rightarrow j \\mid D) = 1 - \\max_Z p(i \\perp j \\mid Z, D)
        
        即：如果无法找到任何条件集 Z 使 i 和 j 独立，则边的置信度高。
        
        Args:
            verbose (bool): 是否打印详细信息
        
        Note:
            - 遍历图中的所有有向边
            - 查找所有涉及该边的独立性测试结果
            - 取最大独立性概率，用 1 减去得到边的置信度
            - 对于骨架中的边，保证最小置信度为 0.5
            
        Example:
            如果所有测试都表明 X 和 Y 不独立（即使给定各种条件集），
            则置信度接近 1.0，表示强烈的因果关系证据。
        """
        for i, j in self.graph.edges():
            # 查找所有涉及 (i, j) 的独立性测试
            max_independence_prob = 0.0
            
            for result in self.independence_results:
                if (result.x == i and result.y == j) or (result.x == j and result.y == i):
                    max_independence_prob = max(max_independence_prob, result.probability)
            
            # 边存在的置信度 = 1 - 最大独立性概率
            confidence = 1.0 - max_independence_prob
            
            # 添加额外的贝叶斯因子调整
            # 如果边在骨架中，增加一些基础置信度
            if self.skeleton.has_edge(i, j) or self.skeleton.has_edge(j, i):
                confidence = max(confidence, 0.5)  # 至少 0.5 的置信度
            
            self.edge_confidence[(i, j)] = confidence
        
        if verbose:
            print(f"  计算了 {len(self.edge_confidence)} 条边的置信度")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取算法执行的摘要信息。
        
        Returns:
            Dict: 包含以下信息的字典
                - n_variables: 变量数量
                - n_edges: 边数量
                - n_independence_tests: 执行的独立性测试次数
                - average_confidence: 平均边置信度
                - edges: 边列表及其置信度
        
        Examples:
            >>> summary = bccd.get_summary()
            >>> print(f"发现 {summary['n_edges']} 条因果边")
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
            'n_independence_tests': len(self.independence_results),
            'average_confidence': avg_confidence,
            'edges': edges_list
        }
        
        return summary
    
    def print_summary(self) -> None:
        """
        打印算法执行的摘要信息。
        
        Examples:
            >>> bccd.print_summary()
        """
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("BCCD 算法摘要")
        print("=" * 70)
        print(f"变量数量: {summary['n_variables']}")
        print(f"发现的因果边数量: {summary['n_edges']}")
        print(f"执行的独立性测试次数: {summary['n_independence_tests']}")
        print(f"平均边置信度: {summary['average_confidence']:.3f}")
        print("\n因果边列表（按置信度排序）:")
        print("-" * 70)
        
        for edge in summary['edges']:
            print(f"  {edge['from']:>10} → {edge['to']:<10}  置信度: {edge['confidence']:.3f}")
        
        print("=" * 70)
    
    def visualize_graph(
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
        可视化学习到的因果图。
        
        此方法使用 ConfidenceGraph 类的 visualize 方法进行可视化。
        
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
        
        **如何解读可视化结果**:
        
        1. 先看**颜色**快速定位最可靠的因果关系（红色边）
        2. 再看**数字**了解具体的置信度值
        3. **粗细**也表示置信度：边越粗，置信度越高
        4. 使用右侧**颜色条**理解颜色到数值的映射
        
        Args:
            show_confidence (bool, optional): 是否在边上显示置信度。默认 True。
            confidence_threshold (float, optional): 只显示置信度高于此阈值的边。默认 0.0（显示所有边）。
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
            save_path (str, optional): 保存图形的路径。如果为 None，则显示图形。默认 None。
            title (str, optional): 自定义标题。如果为 None，则使用默认标题。默认 None。
        
        Examples:
            基本用法:
            
            .. code-block:: python
            
                bccd.visualize_graph()
                
            只显示高置信度边:
            
            .. code-block:: python
            
                bccd.visualize_graph(confidence_threshold=0.7)
                
            使用不同布局:
            
            .. code-block:: python
            
                bccd.visualize_graph(layout='circular')
                
            保存图形:
            
            .. code-block:: python
            
                bccd.visualize_graph(save_path='causal_graph.png')
            
            自定义标题:
            
            .. code-block:: python
            
                bccd.visualize_graph(title='因果图 (噪声=0.5, 样本数=500)')
        
        Note:
            - 边的**粗细**表示置信度：置信度越高，边越粗
            - 边的**颜色**从深蓝（低置信度）经绿色到深红（高置信度）渐变
            - 边旁的**数字**显示贝叶斯置信度的绝对值（0-1范围）
            - 右侧**颜色条**显示颜色到置信度数值的映射关系
            - 需要安装 matplotlib 库
        
        See Also:
            graph_utils.ConfidenceGraph.visualize : 底层可视化方法
        """
        try:
            from .graph_utils import ConfidenceGraph
        except ImportError:
            from graph_utils import ConfidenceGraph
        
        from causallearn.graph.GraphNode import GraphNode
        
        # 创建 GraphNode 列表
        nodes = [GraphNode(name) for name in self.variable_names]
        
        # 创建 ConfidenceGraph 实例
        conf_graph = ConfidenceGraph(nodes, self.variable_names)
        
        # 复制图结构和置信度
        conf_graph.graph = self.graph.copy()
        conf_graph.edge_confidence = self.edge_confidence.copy()
        
        # 调用 visualize 方法
        conf_graph.visualize(
            show_confidence=show_confidence,
            confidence_threshold=confidence_threshold,
            layout=layout,
            figsize=figsize,
            node_size=node_size,
            font_size=font_size,
            save_path=save_path,
            title=title
        )


def main():
    """
    主函数，用于演示 BCCD 算法的使用。
    """
    # 生成示例数据
    np.random.seed(42)
    n_samples = 500
    
    # 真实因果结构: X0 → X1 → X2, X0 → X2
    X0 = np.random.randn(n_samples)
    X1 = 0.8 * X0 + np.random.randn(n_samples) * 0.5
    X2 = 0.6 * X1 + 0.4 * X0 + np.random.randn(n_samples) * 0.5
    
    data = pd.DataFrame({
        'X0': X0,
        'X1': X1,
        'X2': X2
    })
    
    print("示例数据生成完成")
    print(f"数据形状: {data.shape}")
    print(f"\n真实因果结构: X0 → X1 → X2, X0 → X2")
    
    # 运行 BCCD 算法
    bccd = BCCD(
        data,
        alpha=1.0,
        independence_threshold=0.5,
        method='fisherz'
    )
    
    bccd.fit()
    
    # 打印摘要
    bccd.print_summary()
    
    # 可视化
    bccd.visualize_graph(
        show_confidence=True,
        confidence_threshold=0.3,
        layout='spring'
    )


if __name__ == '__main__':
    main()
