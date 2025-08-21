import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Node, Tree, Query
from diff_utils import diff,   syn
from typing import List,Set,Tuple
import re
class Formatter:
    def __init__(self):
        self.CPP_LANGUAGE = Language(tspy.language())
        self.parser = Parser(self.CPP_LANGUAGE)
    def __filter_functions(self,root_node:Tree, preserve_funcs:Set[str],keep_func_signature:bool=False):
        query_str="""
        (function_definition 
            name: (identifier)
        )@func_node
        """
        query = Query(self.CPP_LANGUAGE, query_str)
        captures = query.captures(root_node)
        if not captures:
            return None
        func_nodes = captures["func_node"]
        delete_hunks=[]
        for func_node in func_nodes:
            func_name = func_node.child_by_field_name("name").text.decode()
            if func_name not in preserve_funcs:
                delete_hunks.append((func_node.start_point.row,func_node.end_point.row))
        if not delete_hunks:
            return None
        delete_hunks = sorted(delete_hunks, key=lambda x: x[0])
        def merge_hunks(input_hunks:List[Tuple[int]])->bool:
            merged = [input_hunks[0]]

            for start, end in input_hunks[1:]:
                last_start, last_end = merged[-1]
                if start <= last_end:  # 如果区间有交集，合并
                    merged[-1] = (last_start, max(last_end, end))
                else:  # 否则直接加入
                    merged.append((start, end))

            return merged

        delete_hunks = merge_hunks(delete_hunks)
        return delete_hunks
            
        
                
        
# Need changes to make it match by both func_name and params
    def __find_func_nodes(self,root_node:Tree, func_names: Set[str]):
        if not func_names:
            return []
        query_str=f"""
        (function_definition 
            name: (identifier) @func_name
            (#any-of? @func_name {" ".join(func_names)})
        )@func_node
        """
        query = Query(self.CPP_LANGUAGE, query_str)
        captures = query.captures(root_node)
        if not captures:
            return None
        func_node = captures['func_node']
        return func_node
    
    
    def __get_callees(self,root_node: Tree,founded_callees:Set[str])->Set[str]:
        # 遍历目标函数的子节点，寻找函数调用
        query_str = f"""
        (call
            function: (identifier) @callee_name_node
        )
        """
        query = Query(self.CPP_LANGUAGE, query_str)
        captures = query.captures(root_node)
        if not captures:
            return set()
        callees = {name_node.text.decode() for name_node in captures['callee_name_node']}
        for callee in callees:
            if callee in founded_callees:
                continue
            founded_callees.add(callee)
            func_node = self.__find_func_nodes(root_node,set(callee))
            if func_node:
                func_node=func_node[0]
                sub_callees = self.__get_callees(func_node,founded_callees)
                founded_callees.update(sub_callees)
            
        return founded_callees
            
    def __get_added_funcs(self,o_root_node:Tree,n_root_node:Tree):
        query_str=f"""
        (function_definition 
            name: (identifier) @func_name
        )
        """
        query = Query(self.CPP_LANGUAGE, query_str)
        o_captures = query.captures(o_root_node)
        if not o_captures:
            o_funcs=[]
        else:
            o_funcs = [func_name.text.decode() for func_name in o_captures['func_name']]
        n_captures = query.captures(n_root_node)
        if not n_captures:
            n_funcs=[]
        else:
            n_funcs = [n_func_name.text.decode() for n_func_name in n_captures['func_name']]
        o_funcs=set(o_funcs)
        n_funcs=set(n_funcs)
        return n_funcs-o_funcs
    def __get_modified_funcs(self,o_root_node:Tree,patch:str,context_len_offset:int=3):
        # 使用正则表达式匹配patch中的hunk头部信息
        hunk_header_pattern = re.compile(r'\@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
        matches = list(hunk_header_pattern.finditer(patch))
        hunks=[]
        for match in matches:
            old_start, old_count, _, _ = match.groups()
            old_count = old_count or '1'
            old_end = int(old_start) + int(old_count) - context_len_offset - 1
            old_start = int(old_start)+context_len_offset
            hunks.append((old_start,old_end))
        hunks=sorted(hunks, key=lambda x: x[0])
        modified_funcs = set()
        query_str="""
        (function_definition 
            name: (identifier)
        )@func_node
        """
        query = Query(self.CPP_LANGUAGE, query_str)
        captures = query.captures(o_root_node)
        if not captures:
            return set()
        func_nodes = captures['func_node']
        
        for func_node in func_nodes:
            func_name = func_node.child_by_field_name("name").text.decode()
            func_start_row = func_node.start_point.row+1
            func_end_row = func_node.end_point.row+1
            # 判断函数是否在修改范围内
            for hunk_start,hunk_end in hunks:
                if not (hunk_start>func_end_row or hunk_end<func_start_row):
                    modified_funcs.add(func_name)
                    break
                
        return modified_funcs

    def __delete_useless_code(self,origin_code:str,root_node:Tree,preserve_funcs:set)->str:
        if not preserve_funcs:
            return origin_code
        preserve_function_nodes = self.__find_func_nodes(root_node, preserve_funcs) or []       
        
        for preserve_function_node in preserve_function_nodes:
            callees = self.__get_callees(preserve_function_node,set())
            preserve_funcs.update(callees)
        delete_hunks=self.__filter_functions(root_node,preserve_funcs)
        if not delete_hunks:
            return ""
        origin_lines = origin_code.splitlines()
        patch = "--- example.txt\n+++ example.txt"
        deleted_lines_num=0
        for (start_row,end_row) in delete_hunks:
            affected_rows = end_row-start_row+1
            patch += f"\n@@ -{start_row+1},{affected_rows} +{start_row+1-deleted_lines_num},0 @@"
            deleted_lines_num+=affected_rows
            for i in range(start_row,end_row+1):
                patch+=f"\n-{origin_lines[i]}"
        patched_code =  syn(origin_code,patch)
        if not patched_code:
            return ""
        return patched_code
    def format(self, new_code:str, old_code:str, patch:str):
        """
        根据源文件和patch生成修改后文件
        去除源文件和修改后文件中，未被修改的函数并返回
        """
        if not old_code:
            return "", ""
        # 创建 Clang 索引和tree-sitter parser   
        o_tree = self.parser.parse(old_code.encode('utf-8'))
        o_root_node = o_tree.root_node
        n_tree = self.parser.parse(new_code.encode('utf-8'))
        n_root_node = n_tree.root_node
        preserve_funcs = set()
        
        added_funcs = self.__get_added_funcs(o_root_node, n_root_node)
        preserve_funcs.update(added_funcs)

        modified_funcs = self.__get_modified_funcs(o_root_node, patch, context_len_offset=0)
        preserve_funcs.update(modified_funcs)
        
        cleaned_origin_code = self.__delete_useless_code(old_code, o_root_node, preserve_funcs)
        cleaned_new_code = self.__delete_useless_code(new_code, n_root_node, preserve_funcs)
        return cleaned_origin_code, cleaned_new_code