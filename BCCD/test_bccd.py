import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from BCCD import BCCD


# 全局配置：是否显示因果图
SHOW_GRAPHS = True  # 设置为 True 显示图形，False 不显示


class DataGen:
    def __init__(self, n=2000, seed=42):
        self.n = n
        self.seed = seed
        
    def noise(self, scale):
        return np.random.randn(self.n) * scale
    
    def chain(self, noise):
        np.random.seed(self.seed)
        A = self.noise(1.0)
        B = 0.8 * A + self.noise(noise)
        C = 0.7 * B + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C}), "Chain"
    
    def fork(self, noise):
        np.random.seed(self.seed)
        C = self.noise(1.0)
        A = 0.8 * C + self.noise(noise)
        B = 0.7 * C + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C}), "Fork"
    
    def collider(self, noise):
        np.random.seed(self.seed)
        A = self.noise(1.0)
        B = self.noise(1.0)
        C = 0.7 * A + 0.6 * B + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C}), "Collider"
    
    def triangle(self, noise):
        np.random.seed(self.seed)
        A = self.noise(1.0)
        B = 0.8 * A + self.noise(noise)
        C = 0.5 * A + 0.5 * B + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C}), "Triangle"
    
    def chain4(self, noise):
        np.random.seed(self.seed)
        A = self.noise(1.0)
        B = 0.8 * A + self.noise(noise)
        C = 0.7 * B + self.noise(noise)
        D = 0.6 * C + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D}), "Chain4"
    
    def diamond(self, noise):
        np.random.seed(self.seed)
        A = self.noise(1.0)
        B = 0.8 * A + self.noise(noise)
        C = 0.7 * A + self.noise(noise)
        D = 0.6 * B + 0.5 * C + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D}), "Diamond"
    
    def complex(self, noise):
        np.random.seed(self.seed)
        A = self.noise(1.0)
        B = 0.8 * A + self.noise(noise)
        C = 0.7 * A + self.noise(noise)
        D = 0.5 * B + 0.6 * C + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D}), "Complex"
    
    def vars5(self, noise):
        np.random.seed(self.seed)
        A = self.noise(1.0)
        B = 0.8 * A + self.noise(noise)
        C = 0.7 * B + self.noise(noise)
        D = 0.6 * B + self.noise(noise)
        E = 0.5 * C + self.noise(noise)
        return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E}), "5Vars"


def run_test(tid, data, name, noise, visualize=False):
    print(f"\n{'='*60}")
    print(f"Test {tid}: {name} | Noise={noise:.2f}")
    print(f"Samples={len(data)} | Vars={len(data.columns)}")
    print(f"{'='*60}")
    
    model = BCCD(data, alpha=1.0, independence_threshold=0.5, method='fisherz')
    model.fit(verbose=False)
    
    summary = model.get_summary()
    print(f"Edges: {summary['n_edges']} | Avg. Confidence: {summary['average_confidence']:.3f}")
    
    if summary['edges']:
        print("Top edges:")
        for e in summary['edges'][:5]:
            print(f"  {e['from']} -> {e['to']} ({e['confidence']:.3f})")
    
    # 显示因果图
    if visualize:
        try:
            # 构造包含超参数的动态标题
            custom_title = (
                f'因果图: {name} | 噪声={noise:.2f} | 样本数={len(data)}\n'
                f'变量数={len(data.columns)} | 边数={summary["n_edges"]} | '
                f'平均置信度={summary["average_confidence"]:.3f}'
            )
            model.visualize_graph(
                show_confidence=True,
                confidence_threshold=0.0,
                layout='spring',
                figsize=(10, 8),
                node_size=2000,
                font_size=11,
                title=custom_title
            )
            plt.pause(0.1)
        except Exception as e:
            print(f"警告: 显示图形失败: {e}")
    
    return {
        'id': tid,
        'name': name,
        'noise': noise,
        'edges': summary['n_edges'],
        'conf': summary['average_confidence']
    }


def main():
    print("BCCD Test Suite")
    print("="*60)
    
    gen = DataGen(n=500, seed=42)
    
    configs = [
        (gen.chain, 'Chain'),
        (gen.fork, 'Fork'),
        (gen.collider, 'Collider'),
        (gen.triangle, 'Triangle'),
        (gen.chain4, 'Chain4'),
        (gen.diamond, 'Diamond'),
        (gen.complex, 'Complex'),
        (gen.vars5, '5Vars')
    ]
    
    noise_levels = [0.3, 0.8, 2.5]
    
    results = []
    tid = 1
    
    for func, name in configs:
        for noise in noise_levels:
            gen.seed = 42 + tid
            data, label = func(noise)
            # 使用全局配置控制是否显示图形
            result = run_test(tid, data, label, noise, visualize=SHOW_GRAPHS)
            results.append(result)
            tid += 1
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'ID':<4} {'Name':<10} {'Noise':<7} {'Edges':<6} {'Avg. Conf':<7}")
    print(f"{'-'*60}")
    
    for r in results:
        print(f"{r['id']:<4} {r['name']:<10} {r['noise']:<7.2f} {r['edges']:<6} {r['conf']:<7.3f}")
    
    print(f"{'='*60}")
    print(f"Total: {len(results)} tests")
    

if __name__ == '__main__':
    main()
