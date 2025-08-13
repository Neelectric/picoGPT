from math import sqrt
import torch


class picoSelfAttention:
    def __init__(self, d_hidden):
        self.d_hidden = torch.tensor(d_hidden)
        self.w_query = torch.nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=False)
        self.w_keys = torch.nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=False)
        self.w_values = torch.nn.Linear(in_features=d_hidden, out_features=d_hidden, bias=False)
        
        self.w_query.weight.data = torch.tensor([[-0.381, -0.354],
                                            [ 0.407, -0.601]], requires_grad=True)
        self.w_keys.weight.data = torch.tensor([[ 0.353, -0.159],
                                           [ 0.429, -0.679]], requires_grad=True)
        self.w_values.weight.data = torch.tensor([[0.231, 0.626],
                                      [0.561, 0.024]], requires_grad=True)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, hidden_states):
        queries = self.w_query(hidden_states)
        print(f"queries (ie hidden_states * W_q) are:\n{queries}")
        keys = self.w_keys(hidden_states)
        values = self.w_values(hidden_states)
        print(f"We have shape of queries: {queries.shape}")
        print(f"We have shape of keys.mT: {keys.mT.shape}")
        
        matmul = torch.matmul(queries, keys.mT)
        sqrted = torch.sqrt(self.d_hidden)
        
        print(f"Result of matmulled is {matmul}")
        print(f"Result of sqrted is {sqrted}")
        print(self.d_hidden)
        
        attention_map = matmul / sqrted
        attention_map_sm = self.softmax(attention_map)
        attention = attention_map_sm * values
        return hidden_states
    




class picoGPT:
    def __init__(self):
        pass
    
    
# hidden_states = torch.randn(2, 3, 2)
hidden_states = torch.tensor(
    # [
    [[-0.083, 0.147],
     [0.029, 0.008],
     [-0.204, 0.132]],
    # [[-0.112, -0.081],
    #  [0.238, 0.116],
    #  [0.090, -0.034]],
    # ]
    )
print(f"our hidden states have shape {hidden_states.shape}")
print(f"hidden states are:\n{hidden_states}")
self_attn = picoSelfAttention(d_hidden=2)
print(f"query weights are:\n{self_attn.w_query.weight}")
self_attn.forward(hidden_states)