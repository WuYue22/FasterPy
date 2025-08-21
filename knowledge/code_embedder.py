# This file is for the code to vector function
import torch
from .unixcoder import UniXcoder
import torch.nn.functional as F
import numpy as np

class CodeEmbedder:
    def __init__(self, model_name:str = "./unixcoder-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniXcoder(model_name)
        self.model.to(self.device)
     

    def __call__(self, code: str, keep_tensor: bool=False) -> list | torch.Tensor:
        tokens_ids = self.model.tokenize([code],mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(self.device)
        _,code_embedding = self.model(source_ids)
        if keep_tensor:
            return code_embedding
        normalized_code_embedding = F.normalize(code_embedding, p=2, dim=1).flatten().cpu().detach().tolist()
        return normalized_code_embedding  # 获取代码表示向量


    def cal_similarity(self, cpp_code_1:str, cpp_code_2:str) -> float:
        # 计算代码片段的向量表示
        ce = CodeEmbedder()
        v1 = ce(cpp_code_1,keep_tensor=True)  # 取第一个
        v2 = ce(cpp_code_2,keep_tensor=True)  # 取第二个
        # 计算余弦相似度
        # Normalize embedding
        v1 = F.normalize(v1, p=2, dim=1)
        v2 = F.normalize(v2, p=2, dim=1)
        similarity = torch.einsum("ac,bc->ab",v1,v2).cpu().detach().numpy()[0][0]
        return similarity

def cal_similarity_by_vec(vector_1:torch.Tensor, vector_2:torch.Tensor):
    return torch.einsum("ac,bc->ab",vector_1,vector_2).cpu().detach().numpy()[0][0]



if __name__ == '__main__':
    ce = CodeEmbedder()
    code1 = """
    #include <iostream>

void reverseString(std::string &s) {
    int left = 0, right = s.length() - 1;
    while (left < right) {
        std::swap(s[left], s[right]);
        left++;
        right--;
    }
}

int main() {
    std::string str = "hello";
    reverseString(str);
    std::cout << str << std::endl;  // 输出: "olleh"
    return 0;
}

    """
    code2 = """
    #include <iostream>
#include <algorithm>

void reverseString(std::string &s) {
    std::reverse(s.begin(), s.end());
}

int main() {
    std::string str = "hello";
    reverseString(str);
    std::cout << str << std::endl;  // 输出: "olleh"
    return 0;
}

    """
    ip_similarity = ce.cal_similarity(code1, code2)
    # 输出相似度
    print(f"代码克隆相似度: {ip_similarity:.4f}")
