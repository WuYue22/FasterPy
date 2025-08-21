import subprocess
from typing import Generator
import re
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node, Tree, Query
import clang.cindex
cursor_kind = clang.cindex.CursorKind
Cursor = clang.cindex.Cursor

class Point:
    def __init__(self,row:int,column:int):
        self.row = row
        self.column=column
    def __eq__(self, other):
        return isinstance(other, Point) and self.row == other.row and self.column == other.column
    def __hash__(self):
        return hash((self.row, self.column))

class Function_Definition:
    def __init__(self, name: str, params: list,start_point:Point=None,end_point:Point=None):
        self.name = name
        self.params = tuple(params)
        self.start_point = start_point
        self.end_point = end_point
        
    def __eq__(self, other):
        return isinstance(other, Function_Definition) and self.name == other.name and self.params == other.params

    def __hash__(self):
        return hash((self.name, self.params))  # 用元组计算哈希值
    
    def in_list(self, fd_list: list):
        for fd in fd_list:
            if self==fd:
                return True
        return False

class Formatter:
    def __init__(self):
        self.index = clang.cindex.Index.create()
        self.CPP_LANGUAGE = Language(tscpp.language())
        self.parser = Parser(self.CPP_LANGUAGE)


# tree-sitter cpp code ****************************************************

    def __get_undefined_classes(self,node: Tree):
        """
        解析 C++ 代码，找出函数声明中使用但未定义的类
        """

        defined_classes = set()  # 已定义的类
        used_classes = set()  # 在函数签名中使用的类

        # 遍历 AST 查找已定义的类
        def traverse(node):
            if node.type == "class_specifier":
                class_name = node.child_by_field_name("name")
                if class_name:
                    defined_classes.add(class_name.text.decode("utf-8"))

            elif node.type == "function_declarator":
                # 查找返回类型
                type_node = node.child_by_field_name("type")
                if type_node and type_node.type == "type_identifier":
                    used_classes.add(type_node.text.decode("utf-8"))

                # 查找参数类型
                parameters_node = node.child_by_field_name("parameters")
                if parameters_node:
                    for param in parameters_node.children:
                        type_node = param.child_by_field_name("type")
                        if not type_node:
                            continue
                        type = type_node.type
                        if type == "type_identifier":
                            used_classes.add(type_node.text.decode("utf-8"))
                                
            # 递归遍历子节点
            for child in node.children:
                traverse(child)

        traverse(node)

        # 计算未定义的类
        undefined_classes = used_classes - defined_classes
        return undefined_classes

    def __add_missing_classes(self,source_code:str,tc_root_node: Tree):
        """
        在 C++ 代码前补充未定义的类
        """
        undefined_classes = self.__get_undefined_classes(tc_root_node)
        class_definitions = " ".join(f"class {cls} {{}};" for cls in undefined_classes)

        if class_definitions:
            new_source_code = class_definitions + "\n" + source_code
            return new_source_code
        return source_code

    def format_cpp_code(self,code: str, style="llvm") -> str:
        process = subprocess.Popen(
            ["clang-format", f"--style={style}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        formatted_code, error = process.communicate(code)
        if error:
            raise RuntimeError(f"clang-format error: {error}")
        return formatted_code
    def __traverse_tree(self, tree: Tree) -> Generator[Node, None, None]:
        cursor = tree.walk()

        visited_children = False
        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break


    def __get_params(self, param_list_node: Tree):
        query_str = """
        ( parameter_declaration type: (_) @param_type )
        """
        query = Query(self.CPP_LANGUAGE, query_str)
        captures = query.captures(param_list_node)
        if not captures:
            return []
        param_type_nodes=captures['param_type']
        return [param_type_node.text.decode("utf-8") for param_type_node in param_type_nodes]

    def __filter_functions(self,node:Tree, preserve_func_list:list, skip_func_body:bool=False):
        if node.type == "comment":  # 跳过注释
            return ""
        elif node.type == "function_definition":
            # 如果函数名在保留列表中，则保留函数
            function_declarator = node.child_by_field_name("declarator")
            function_name = function_declarator.child_by_field_name("declarator").text.decode("utf-8")
            params = self.__get_params(function_declarator.child_by_field_name("parameters"))
            fd = Function_Definition(function_name,params)
            if not fd.in_list(preserve_func_list):
                return " ".join(self.__filter_functions(child,[],True) for child in node.children)
        elif node.type == "compound_statement" and skip_func_body:
            return "{}"
        
        children_text = " ".join(self.__filter_functions(child,preserve_func_list) for child in node.children)  # 递归处理子节点
        return node.text.decode() if not node.children else children_text  # 如果有子节点，则拼接
    
    def get_all_func_defintion(self, code:str):
        query_str = f"""
        (function_definition
            declarator: (function_declarator 
                declarator: (identifier) @function_name
                parameters: (parameter_list) @param_list 
            ) 
        ) @func_definition
        """
        tree = self.parser.parse(code.encode('utf-8'))
        tc_root_node = tree.root_node
        query = Query(self.CPP_LANGUAGE, query_str)
        captures = query.captures(tc_root_node)
        if not captures:
            return []
        function_definitions = captures['func_definition']
        function_names = captures['function_name']
        param_list_nodes = captures['param_list']
        return [Function_Definition(function_node.text.decode('utf-8'),
                                    self.__get_params(param_list_node),
                                    Point(function_definition.start_point.row,function_definition.start_point.column),
                                    Point(function_definition.end_point.row,function_definition.end_point.column)
                                    ) 
                for function_node, param_list_node,function_definition in zip(function_names, param_list_nodes,function_definitions)]

    # def __filter_funtions2(self,node:Tree, preserve_func_list:list):
        
    
    def __get_func_defintion(self, root_node: Tree, function_name: str):
        query_str = f"""
        (function_definition
            type: (_) @return_type
            declarator: (function_declarator 
                declarator: (identifier) @function_name
                (#eq? @function_name {function_name})
                parameters: (parameter_list) @param_list 
            ) 
        )
        """
        query = Query(self.CPP_LANGUAGE, query_str)
        captures = query.captures(root_node)
        if not captures:
            return []
        function_nodes = captures['function_name']
        # return_nodes = captures['return_type']
        param_list_nodes = captures['param_list']
        return [Function_Definition(function_node.text.decode('utf-8'),self.__get_params(param_list_node)) for function_node, param_list_node in zip(function_nodes, param_list_nodes)]
        # for function_node, return_node, param_list_node in zip(function_nodes, return_nodes, param_list_nodes):
        #     return_type = return_node.text.decode("utf-8")
        #     function_name = function_node.text.decode("utf-8")
        #     param_types = get_params(param_list_node)
        #     print(f"function name:{function_name}, return type: {return_type}, param_types: {param_types}")
        
    # clang parser *****************************************************

    def __get_callees(self,node: Cursor, tc_root_node: Tree, match_function_name: bool = False):
        # 遍历目标函数的子节点，寻找函数调用
        callees = set()
        for child in node.get_children():
            if child.kind == clang.cindex.CursorKind.CALL_EXPR:
                callee_node = child.get_definition()
                if callee_node:
                    func_name_params=callee_node.displayname
                    result = re.split(r'[(), ]', func_name_params)
                    # 去除空字符串
                    result = [s for s in result if s]
                    func_name = result[0]
                    if len(result)>1:
                        params = result[1:]
                    else:
                        params = []
                    callees.add(Function_Definition(func_name,params))
                    sub_callees = self.__get_callees(callee_node,tc_root_node,match_function_name)
                    callees.update(sub_callees)
                elif match_function_name:
                    # CursorKind.DECL_REF_EXPR->CursorKind.OVERLOADED_DECL_REF
                    func_name=list(list(child.get_children())[0].get_children())[0].displayname
                    possible_references = self.__get_func_defintion(tc_root_node, func_name)
                    if possible_references:
                        # 只取第一个匹配的函数
                        callees.add(possible_references[0])
                    

                
            else:
                sub_callees = self.__get_callees(child,tc_root_node,match_function_name)
                if sub_callees:
                    callees.update(sub_callees)
                # if called_function and called_function.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                #     print(f"  Calls function: {called_function.spelling} at {child.location.file}:{child.location.line}")
                #     if called_function.location.file:
                #         print(f"    Defined at: {called_function.location.file}:{called_function.location.line}")
                #     else:
                #         print(f"    Function definition not found (might be an external function)")
        return callees
            
    # Need changes to make it match by both func_name and params
    def __find_func_node(self,root_node, func_name: str):
        for node in root_node.get_children():
            if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and node.spelling == func_name:
                return node
            else:
                result = self.__find_func_node(node, func_name)
                if result:
                    return result
    
    # whole format process
    def format(self, origin_code: str, target_function: str, match_function_name:bool=False):
        """
        解析源代码并生成AST
        """
        
        # 创建 Clang 索引和tree-sitter parser   
        tree = self.parser.parse(origin_code.encode('utf-8'))
        tc_root_node = tree.root_node
        updated_code = self.__add_missing_classes(origin_code,tc_root_node)
        tu = self.index.parse("fake.cpp", args=['-x', 'c++', '-std=c++17','-I/usr/local/include','-stdlib=libc++'],unsaved_files=[("fake.cpp", updated_code)])

        node = tu.cursor
        target_function_node = self.__find_func_node(node, target_function)
        if not target_function_node:
            print(f"Function {target_function} not found")
            return
        
        
        # 遍历AST获取callee
        callees = self.__get_callees(target_function_node,tc_root_node, match_function_name)
        return self.format_cpp_code(self.__filter_functions(tc_root_node, callees))
        
                
        
        

    def print_ast(self,node, depth=0):
        print("  " * depth + f"{node.kind} ({node.spelling})")
        for child in node.get_children():
            self.print_ast(child, depth + 1)


if __name__ == "__main__":
    formatter = Formatter()
    origin_code = None
    with open("example.cpp", 'r', encoding='utf-8') as f:
        origin_code = f.read()
    formatted_code = formatter.format(origin_code, "main", match_function_name=False)
    print(formatted_code)