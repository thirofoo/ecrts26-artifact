import os
import networkx as nx
from src.generator.wrapper import RDGenWrapper
from src.generator.converter import MCDAGConverter, MCConfig
from src.analyzer.multipath import MultipathAnalyzer
from src.common.data_manager import DataManager
from src.utils.visualizer import draw_dag

def main():
    # --- Configuration ---
    config_path = "src/generator/config_templates/test_config.yaml"
    output_dir = "data/raw_dags/test_run"
    save_path = "data/processed/dag_dataset.pkl"
    
    # Plot output paths.
    plot_path_hi = "data/results/example_task_hi.png"
    plot_path_lo = "data/results/example_task_lo.png"
    
    # Failure rate used for visualization labels.
    VISUALIZATION_FAILURE_RATE = 1e-9 
    
    mc_config = MCConfig(
        force_criticality="HI",
        p_hi=0.5, 
        factor_min=2.0,
        factor_max=4.0,
        deadline_type="implicit",
        deadline_slack_min=1.2,
        deadline_slack_max=1.5
    )
    # ------------

    wrapper = RDGenWrapper()
    converter = MCDAGConverter(mc_config)

    try:
        # 1. Generate DAGs and inject MC parameters.
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            
        xml_path = wrapper.run(config_path, output_dir)
        raw_dags = wrapper.parse_xml(xml_path)
        
        if not raw_dags: return
        mc_dags = converter.convert(raw_dags)

        # 2. Save data.
        DataManager.save(mc_dags, save_path)
        print(f"Generated {len(mc_dags)} DAGs with Criticality Mode: {mc_config.force_criticality}")

        # 3. Visualize examples.
        
        # --- (A) HI Task Example ---
        hi_tasks = [d for d in mc_dags if d.criticality == "HI"]
        if hi_tasks:
            target_dag = hi_tasks[0]
            
            w_hi = MultipathAnalyzer.get_workload(target_dag, "HI")
            u_hi = w_hi / target_dag.period if target_dag.period > 0 else 0
            
            total_prob = target_dag.get_total_failure_prob(VISUALIZATION_FAILURE_RATE)
            
            print(f"\n--- Visualizing HI Task: {target_dag.id} ---")
            print(f"U_HI: {u_hi:.3f}, T: {target_dag.period}, D: {target_dag.deadline}")
            print(f"Total Failure Prob: {total_prob:.2e}")

            G = nx.DiGraph()
            colors = {}
            labels = {}
            for node in target_dag.nodes.values():
                G.add_node(node.id)
                colors[node.id] = "salmon"
                
                p_node = node.get_failure_prob(VISUALIZATION_FAILURE_RATE)
                
                labels[node.id] = f"{node.id}\n{node.c_lo}->{node.c_hi}\nP={p_node:.1e}"
                
                for succ in node.successors:
                    G.add_edge(node.id, succ)
            
            title = (f"HI-Criticality Task (ID: {target_dag.id})\n"
                     f"U_HI: {u_hi:.2f} | P_total: {total_prob:.2e}")
            draw_dag(G, target_dag, title, plot_path_hi, colors, labels)
        else:
            print("No HI tasks generated (Check config).")

        # --- (B) LO Task Example ---
        lo_tasks = [d for d in mc_dags if d.criticality == "LO"]
        if lo_tasks:
            target_dag = lo_tasks[0]
            
            w_lo = MultipathAnalyzer.get_workload(target_dag, "LO")
            u_lo = w_lo / target_dag.period if target_dag.period > 0 else 0
            total_prob = target_dag.get_total_failure_prob(VISUALIZATION_FAILURE_RATE)
            
            print(f"\n--- Visualizing LO Task: {target_dag.id} ---")
            
            G = nx.DiGraph()
            colors = {}
            labels = {}
            for node in target_dag.nodes.values():
                G.add_node(node.id)
                colors[node.id] = "lightblue"
                
                p_node = node.get_failure_prob(VISUALIZATION_FAILURE_RATE)
                
                labels[node.id] = f"{node.id}\n{node.c_lo}\nP={p_node:.1e}"
                for succ in node.successors:
                    G.add_edge(node.id, succ)
            
            title = (f"LO-Criticality Task (ID: {target_dag.id})\n"
                     f"U_LO: {u_lo:.2f} | P_total: {total_prob:.2e}")
            draw_dag(G, target_dag, title, plot_path_lo, colors, labels)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
