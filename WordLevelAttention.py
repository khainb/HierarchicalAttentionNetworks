import torch
import torch.nn as nn
import numpy as np
from utils import mat_mul,element_wise_mul
import torch.nn.functional as F
class WordLevelAttention(nn.Module):
    def __init__(self,word_emb,hidden_size):
        super(WordLevelAttention,self).__init__()
        emb_size= word_emb.shape[1]
        #self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_weight = nn.Linear(2*hidden_size,2 *hidden_size)
        #self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        #self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.context_weight = nn.Linear(2*hidden_size,1,bias=False)
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.device=device
        self.lookup = nn.Embedding.from_pretrained(torch.from_numpy(word_emb).to(device),freeze=False)
        self.gru= nn.GRU(emb_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        torch.nn.init.xavier_uniform(self.word_weight.weight)
        torch.nn.init.xavier_uniform(self.context_weight.weight)
        #self.word_weight.data.normal_(mean, std)
        #self.context_weight.data.normal_(mean, std)
        #self.word_bias.data.normal_(mean,std)

    def forward(self, input, hidden_state,sent_len):
        mask = torch.arange(input.shape[0]).expand(len(sent_len),input.shape[0]).to(self.device) < sent_len.unsqueeze(1)
        output = self.lookup(input) #[Seq_len,batch_size,emb_size]
        f_output, h_output = self.gru(output.float(), hidden_state) # f_output [Seq_len,batch_size,2*hidden_size]
        output = self.word_weight(f_output)
        output = self.context_weight(output).squeeze().permute((1,0))
        #output = mat_mul(f_output,self.word_weight,self.word_bias) # out_put [Seq_len,batch_size,2*hidden_size]
        #output = mat_mul(output,self.context_weight).permute((1,0)) # [batch_size,Seq_len]
        #print(torch.isnan(output).any())
        exps = torch.exp(output) * mask.float()
        output =exps / (torch.sum(exps,dim=1).unsqueeze(1)+1e-8) #[batch_size,Seq_len]
        #output[~mask] = float('-inf')
        #output = F.softmax(output,dim=1) #[batch_size,Seq_len]
        output = element_wise_mul(f_output,output.permute((1,0))) # [1,batch_size,2*hidden_size]

        return output, h_output
