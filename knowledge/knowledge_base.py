# This file is for the knowledge base function
from pymilvus import MilvusClient, CollectionSchema
from .code_embedder import CodeEmbedder
import numpy as np

DEFAULT_DB_NAME = "milvus_demo"
DEFAULT_COLLECTION_NAME = "demo_collection"
DEFAULT_DIMENSION = 768

# Data format: {"vector": [0.1, 0.2, 0.3, 0.4, 0.5],"src_code": "source code string","patch":"improve-patch string"}

def l1_norm(data):
    return data / np.sum(np.abs(data))
class KnowledgeBase:
    def __init__(self, db_name:str = DEFAULT_DB_NAME, collection_name:str = DEFAULT_COLLECTION_NAME, dimension:int = DEFAULT_DIMENSION):
        client = MilvusClient(f"{db_name}.db")
        if client.has_collection(collection_name=collection_name):
            # client.drop_collection(collection_name="demo_collection")
            client.load_collection(collection_name=collection_name)
            res = client.get_load_state(collection_name=collection_name)['state']
            # if res != :
            #     raise Exception("Collection load failed")
        else:   
            from pymilvus import DataType, FieldSchema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True,auto_id = True),
                # configure default value `25` for field `age`
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,dim = DEFAULT_DIMENSION),
                # FieldSchema(name="src_code", dtype=DataType.VARCHAR, description="age",max_length=6000),
                FieldSchema(name="src_code_len", dtype=DataType.INT64, description="line nums of src code"),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, description="summary of changes",max_length=300),
                FieldSchema(name="rate", dtype=DataType.FLOAT, description="improving rate"),
            ]
            schema = CollectionSchema(fields=fields,auto_id=True, enable_dynamic_field=False)
            client.create_collection(
                collection_name=collection_name,
                schema=schema
                # dimension=dimension,  # The vectors we will use in this demo has 768 dimensions
            )
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="IP",
                index_type="FLAT",
                index_name="vector_index"
            )

            client.create_index(
                collection_name=collection_name,
                index_params=index_params,
                sync=True # Whether to wait for index creation to complete before returning. Defaults to True.
            )

        self.client = client
        self.ce = CodeEmbedder()
    
    def insert(self, datas:list, collection_name:str = DEFAULT_COLLECTION_NAME):
        datas = [{"vector": self.ce(data["src_code"]), "src_code_len": data["src_code_len"], "summary": data["summary"],"rate":data["rate"]} for data in datas]
        self.client.insert(collection_name=collection_name, data=datas)
    def insert_with_vector(self, datas:list, collection_name:str = DEFAULT_COLLECTION_NAME):
        self.client.insert(collection_name=collection_name, data=datas)
    def insert_single(self, data:dict, collection_name:str = DEFAULT_COLLECTION_NAME):
        data = {"vector": self.ce(data["src_code"]), "src_code_len": data["src_code_len"], "summary": data["summary"],"rate":data["rate"]}
        self.client.insert(collection_name=collection_name, data=data)  
    def insert_single_with_vector(self, data:dict, collection_name:str = DEFAULT_COLLECTION_NAME):
        data = {"vector": data["vector"], "src_code_len": data["src_code_len"], "summary": data["summary"],"rate":data["rate"]}
        self.client.insert(collection_name=collection_name, data=data)

    def _filter(self, search_result:list, top_k:int = 2, distance_range:float=0.2):
        results=[]
        for hits in search_result:
            hits=[hit for hit in hits if hit["distance"] != 1.0]
            if len(hits)<=top_k:
                results.append(hits)
                continue
            sort_by_distance = sorted(hits, key=lambda x: x["distance"], reverse=True)
            __result=[sort_by_distance[0]]
            sort_by_distance = sort_by_distance[1:]
            minimum_distance = sort_by_distance[0]["distance"]-distance_range
            sort_by_distance = [hit for hit in sort_by_distance if hit["distance"] >= minimum_distance]
            if len(sort_by_distance) < top_k-1:
                __result.extend(sort_by_distance)
                continue
            
            sort_by_rates=sorted(hits, key=lambda x: x['entity']["rate"], reverse=True)
            __result.extend(sort_by_rates[:top_k-1])
            results.append(__result)
        
        return results
                
                
    def search(self, query_codes:list=None,query_vector:list=None, rate_thredhold:float=0.1,similarity_thredhold:float=0.5,top_k:int = 2,search_limit=10, collection_name:str = DEFAULT_COLLECTION_NAME, output_fields:list = ["src_code_len","summary","rate"], metric_type:str="IP",filter_fn_on:bool=True):
        if query_codes:
            data=[self.ce(query_code) for query_code in query_codes]
        elif query_vector:
            data=query_vector
        else:
            raise Exception("No query code or vector")
        filter = f"rate > {rate_thredhold}"
        res = self.client.search(collection_name=collection_name, 
                                  data=data, 
                                  limit=search_limit, 
                                  output_fields=output_fields,
                                  filter=filter,
                                  search_params={"metric_type": metric_type}
                                  )
        # filter out datas with distance below threhold
        res = [[hit for hit in hits if hit["distance"] >= similarity_thredhold] for hits in res]
        if filter_fn_on:
            filtered_result= self._filter(res, top_k=top_k)
            return filtered_result
        else:
            return res
        # final_result = [[{'score': hit["score"], "summary": hit["entity"]['summary']} for hit in hits] for hits in filtered_result]
        
        
    
    def drop_collection(self, collection_name:str=DEFAULT_COLLECTION_NAME):
        self.client.drop_collection(collection_name=collection_name)
        
        
if __name__ == "__main__":
        
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
    code2="""
    int useless(){
    // comments
    int a = 1;
    int b = 2;
    int c = a + b;
    printf("c: %d\n", c);
    return c;
}
    """
    code3 = """
D, G = list(map(int, input().split()))
P = []
num_p = 0
for i in range(1, D + 1):
    p, c = list(map(int, input().split()))
    num_p += p
    for j in range(1, p + 1):
        P += [(j, i * 100 * j + c * (j == p))]
dp = [0] * (num_p + 1)
for k, pt in P:
    if k == 1:
        t_dp = dp[:]
    for cur in range(1, num_p + 1):
        if cur >= k:
            dp[cur] = max(dp[cur], t_dp[cur - k] + pt)
for i in range(num_p + 1):
    if dp[i] >= G:
        print(i)
        break


    """
    kb  = KnowledgeBase(db_name="CKB")
    # kb.drop_collection()
    # datas = ({"src_code":code1,"patch":"patch1","rate":0.7},{"src_code":code2,"patch":"patch2","rate":0.3})
    # kb.insert(datas=datas)
    import time
    start_time=time.time()
    result = kb.search([code1,code2,code3]*100)
    print(time.time()-start_time)
    # for hits in result:
    #     for hit in hits:
    #         print(hit,"\n")
    