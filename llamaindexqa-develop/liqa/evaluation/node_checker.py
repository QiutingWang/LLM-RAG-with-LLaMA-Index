import seaborn as sb
import matplotlib.pyplot as plt
import json
class Checker:
        
    @staticmethod
    def parse_plt(nodes,step_ratio:int=10):
        data = [len(node.text) for node in nodes]
        sb.barplot( data)
        plt.xticks(list(range(0, len(data), int(len(data)/step_ratio))))  
        plt.show()
        
    @staticmethod
    def text_length(nodes, length_threshold:int=None, save_to_file:bool=False):

        long_nodes = [node for node in nodes if len(node.text) > length_threshold]
        long_nodes_dict = {node.node_id: node.text for i, node in enumerate(long_nodes)}
        if save_to_file:
            with open('liqa/sample/sample.json', 'w') as f:
                json.dump(long_nodes_dict, f)
        return long_nodes_dict

    @staticmethod
    def empty_metadata(nodes):
        """
        查询一组nodes中metadata为空
        """
        empty_metadata_nodes = [node.id for node in nodes if not node.metadata]
        if empty_metadata_nodes != list():
            print("存在空node",empty_metadata_nodes)
        return empty_metadata_nodes
    
    @staticmethod
    def nodes_id(nodes,node_id):
        return [node for node in nodes if node.node_id==node_id][0]
    @staticmethod
    def nodes_file(nodes,filename):
        return [node for node in  nodes if (node.metadata.get('filename')==filename or node.metadata.get('file_name')==filename)]

