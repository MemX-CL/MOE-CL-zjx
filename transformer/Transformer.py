import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




def get_PCA_obj(inputs, threshold = 0.9):
    # embeddings = [N, embed_dim]
    print('create PCA object')
    inputs = inputs.mean(axis = 1)
    inputs = inputs.cpu().numpy()
    pca = PCA()
    pca.fit(inputs)

    # find the first k components that in total explain threshold of the variance
    unique_dims = 0
    total = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        total += var
        if total >= threshold:
            unique_dims = i
            break

    # print(f'pca unique dims: {unique_dims},  {pca.explained_variance_ratio_[:unique_dims]}')
    # print(torch.tensor(pca.transform(inputs)[:,:unique_dims]).shape)
    return torch.tensor(pca.transform(inputs)[:,:unique_dims]), unique_dims


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
class MHSA(nn.Module):
    def __init__(self, config):
        super(MHSA, self).__init__()
        self.config = config
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        
        self.input_dim = config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(self.input_dim, self.attention_head_size * 3, bias=False) for _ in range(self.num_attention_heads)])
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.head_mask = [1] * self.num_attention_heads
    
    def forward(self, hidden_states: torch.Tensor, attention_mask):
        qkv = torch.stack([self.heads[h](hidden_states) for h in range(self.num_attention_heads)])
        # qkv = torch.stack([self.heads[h](hidden_states) * self.head_mask[h] for h in range(self.num_attention_heads)])
        # batch_size, seq_len, _ = hidden_states.shape
        # qkv = torch.stack([
        #     self.heads[h](hidden_states) if self.head_mask[h] else hidden_states.new_zeros((batch_size, seq_len, self.attention_head_size * 3))
        #     for h in range(self.num_attention_heads)
        # ])
        q, k, v = tuple(rearrange(qkv, 'h b n (k d) -> k b h n d', k=3))

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor
        scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, None, :] == 0, -torch.inf)
        attention = self.softmax(scaled_dot_prod)
        self.attention = attention

        # batch_size, num_head, seq_len, head_dim
        result = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        result = rearrange(result, "b h n d -> b n (h d)")
        return result

class MHSA_SIMPLE(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(MHSA_SIMPLE, self).__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size / num_attention_heads)
        self.num_attention_heads = num_attention_heads
        
        self.input_dim = self.hidden_size
        self.heads = nn.ModuleList([nn.Linear(self.input_dim, self.attention_head_size * 3, bias=False) for _ in range(self.num_attention_heads)])
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.head_mask = [1] * self.num_attention_heads
    
    def forward(self, hidden_states: torch.Tensor, attention_mask):
        qkv = torch.stack([self.heads[h](hidden_states) for h in range(self.num_attention_heads)])
        # qkv = torch.stack([self.heads[h](hidden_states) * self.head_mask[h] for h in range(self.num_attention_heads)])
        # batch_size, seq_len, _ = hidden_states.shape
        # qkv = torch.stack([
        #     self.heads[h](hidden_states) if self.head_mask[h] else hidden_states.new_zeros((batch_size, seq_len, self.attention_head_size * 3))
        #     for h in range(self.num_attention_heads)
        # ])
        q, k, v = tuple(rearrange(qkv, 'h b n (k d) -> k b h n d', k=3))

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor
        scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, None, :] == 0, -torch.inf)
        attention = self.softmax(scaled_dot_prod)
        self.attention = attention

        # batch_size, num_head, seq_len, head_dim
        result = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        result = rearrange(result, "b h n d -> b n (h d)")
        return result
    
    
class Attention_SIMPLE(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads):
        super(Attention_SIMPLE, self).__init__()
        self.self = MHSA_SIMPLE(hidden_size, num_attention_heads) # split multi-head
        # self.self = SelfAttention(config)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):
        hidden_states = self.self(input_tensor, attention_mask)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = MHSA(config) # split multi-head
        # self.self = SelfAttention(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):
        hidden_states = self.self(input_tensor, attention_mask)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class FeedForward_SIMPLE(nn.Module):
    def __init__(self, config, hidden_size):
        super(FeedForward_SIMPLE, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states
    

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states
    
class TransformerEncoderP(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerEncoderP, self).__init__()
        self.attention = Attention(config)

        self.ffn = FeedForward(config)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output_x = ffn_output
        # print(ffn_output.shape)
        # ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output, ffn_output
class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(config)

        self.ffn = FeedForward(config)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)

        # print(ffn_output.shape)
        # ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output
    
class TransformerEncoders(nn.Module):
    def __init__(self, config):
        super(TransformerEncoders, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerEncoder(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
    
    
class LayerDecoder(nn.Module):
    def __init__(self, config):
        super(LayerDecoder, self).__init__()
        self.config = config        
        self.decoder = BertLMPredictionHead(config)
        
    def forward(self, encoder_output) -> torch.Tensor:
        decoder_output = self.decoder(encoder_output)
        return decoder_output
    
    
class BertWithLayerDecoders(nn.Module):
    def __init__(self, config):
        super(BertWithLayerDecoders, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList([TransformerEncoder(config) for _ in range(config.num_hidden_layers)])
        self.decoders = nn.ModuleList([LayerDecoder(config) for _ in range(4)])
    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        main_output = hidden_states
        
        decoder_outputs = []
            
        for i, encoder in enumerate(self.encoders):
            main_output = encoder(main_output, attention_mask)
            if i == 0 or i == 3 or i == 6 or i == 9:
                encoded = main_output.detach()
                decoded = self.decoders[i//3](encoded)
                decoder_outputs.append(decoded)
        
        main_output = [main_output]
        outputs = main_output + decoder_outputs
        
        return outputs
    
   
class BertWithLayerSavers(nn.Module):
    def __init__(self, config):
        super(BertWithLayerSavers, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderP(config) for _ in range(config.num_hidden_layers)])
   
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        layer_outputs = []
        layer_ffn = []
        for i, layer_module in enumerate(self.layers):
            hidden_states, ffn_outputs = layer_module(hidden_states, attention_mask)
            layer_outputs.append(hidden_states)
            layer_ffn.append(ffn_outputs)
        return hidden_states, layer_outputs, layer_ffn
    
class BertLayerSaveModel(nn.Module):
    def __init__(self, config):
        super(BertLayerSaveModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertWithLayerSavers(config)
        
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        outputs, layer_outputs, layer_ffns = self.layers(embeddings, attention_mask)
        return outputs, layer_outputs, layer_ffns
            
    
class BertDecoderModel(nn.Module):
    def __init__(self, config):
        super(BertDecoderModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertWithLayerDecoders(config)
        
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs
    
    
class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoders = TransformerEncoders(config)
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        output = self.encoders(embeddings, attention_mask)
        return output


class BertModelWithMOE(nn.Module):
    def __init__(self, config):
        super(BertModelWithMOE, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoders = TransformerEncoders(config)
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        output, routes = self.encoders(embeddings, attention_mask)
        return output, routes

class MemoryFromDecoder(nn.Module):
    def __init__(self):
        super(MemoryFromDecoder, self).__init__()
    
    def output2input(self, output):
        self.output = output
        self.n = self.output.shape[0]
        self.d = self.output.shape[1]
        self.fake_input = torch.zeros(self.n, self.d, 1)
        softmax_2 = nn.Softmax(dim=2)
        self.output = softmax_2(self.output)

        topks, index_id = torch.topk(self.output, 1, 2)
        for i in range(self.n):
            for j in range(self.d):
                self.fake_input[i][j] = index_id[i][j]

        # for i in range(self.n):
        #     for j in range(self.d):
        #         self.fake_input[i][j] = topks[i][j]

        
        self.fake_input = self.fake_input.to('cuda')
        return self.fake_input
    
class new_expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output

class simple_expert(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads ):
        super().__init__()
        self.attention = Attention_SIMPLE(config, hidden_size, num_attention_heads )
        self.ffn = FeedForward_SIMPLE(config, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output
    

class new_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k = config.num_experts
        self.experts = nn.ModuleList([new_expert(config) for i in range(self.k)])
        
    
    def route(self, hidden_states, cluster_centers):
        sentence_states = torch.mean(hidden_states, dim=1)
        votes ={key:0 for key in range(self.k)}
        
        for sentence_state in sentence_states:
            index =  ((cluster_centers-sentence_state)**2).sum(dim=1).argmin()
            # print(index.item())
            votes[index.item()] += 1

        expert_id = max(votes, key = votes.get)
        return expert_id
    

    def forward(self, hidden_states, attention_mask, center):
        expert_id = self.route(hidden_states, center)
        expert_output = self.experts[expert_id](hidden_states, attention_mask)
        
        return expert_output

class vermilion_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k = config.num_experts
        self.experts = nn.ModuleList([new_expert(config) for i in range(self.k)])
        
    
    def route(self, hidden_states, cluster_centers):
        sentence_states = torch.mean(hidden_states, dim=1)
        # expert_id = []
        # print(sentence_states.shape,cluster_centers.shape)
        distances = torch.cdist(sentence_states.view(sentence_states.shape[0], -1), cluster_centers.view( cluster_centers.shape[0],-1))
        # print(distances.shape)
        nearest = torch.argmin(distances, dim=1)
        # print(nearest)
        expert_id = nearest
        return expert_id
    

    def forward(self, hidden_states, attention_mask, center):
        expert_id = self.route(hidden_states, center)
        output = []
        for j in range(hidden_states.shape[0]):
            # print(attention_mask[j,:].shape)
            
            # expert_output = self.experts[expert_id[j].item()](hidden_states[j,:], attention_mask[j,:])
            
            expert_output = self.experts[expert_id[j].item()](hidden_states[j,:].view(1,self.config.seq_len, self.config.hidden_size), attention_mask[j,:].view(1,self.config.seq_len))
            output.append(expert_output)
        output = torch.cat(output, dim = 0)
        # print(output.shape)
        return output

class simple_layer(nn.Module):
    def __init__(self, config, cluster_num, unique_dim, heads):
        super().__init__()
        self.config = config
        self.unique_dim = unique_dim
        self.k = cluster_num
        self.unique_experts = nn.ModuleList([simple_expert(config, unique_dim, heads) for i in range(self.k)])
        self.common_experts = simple_expert(config, config.hidden_size - unique_dim, heads)
        
    
    def route(self, hidden_states, cluster_centers, pca):
        sentence_states = torch.mean(hidden_states, dim=1)
        sentence_states = torch.tensor(pca.transform(sentence_states.cpu().numpy()))
        unique_states = sentence_states[:, :self.unique_dim]
        # common_states = sentence_states[:, self.unique_dim:]

        votes ={key:0 for key in range(self.k)}
        for sentence_state in unique_states:

            index =  ((cluster_centers.to('cuda')-sentence_state.to('cuda'))**2).sum(dim=1).argmin()
            # print(index.item())
            votes[index.item()] += 1

        expert_id = max(votes, key = votes.get)
        return expert_id
    

    def forward(self, hidden_states, attention_mask, center, pca):
        # n,d,k = hidden_states.shape
        hidden_states = torch.tensor(pca.transform(hidden_states.view(-1,768).cpu().numpy()))
        # print(hidden_states.shape)
        hidden_states = hidden_states.view(-1, self.config.seq_len, self.config.hidden_size).to('cuda')
        unique_states = hidden_states[:, :,:self.unique_dim]
        unique_atten = attention_mask
        common_states = hidden_states[:,:, self.unique_dim:]
        common_atten = attention_mask
        expert_id = self.route(hidden_states, center, pca)
        unique_expert_output = self.unique_experts[expert_id](unique_states, unique_atten)
        common_expert_output = self.common_experts(common_states, common_atten)

        expert_output = torch.cat((unique_expert_output, common_expert_output), dim = 2)
        # print(expert_output.shape)
        return expert_output
class rose_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k = config.num_experts
        self.experts = nn.ModuleList([new_expert(config) for i in range(self.k)])
        
    
    def route(self, hidden_states, cluster_centers, hidden_states_for_route):
        sentence_states = torch.mean(hidden_states_for_route, dim=1)
        # expert_id = []
        # print(sentence_states.shape,cluster_centers.shape)
        distances = torch.cdist(sentence_states.view(sentence_states.shape[0], -1), cluster_centers.view( cluster_centers.shape[0],-1))
        # print(distances.shape)
        nearest = torch.argmin(distances, dim=1)
        # print(nearest)
        expert_id = nearest
        return expert_id
    

    def forward(self, hidden_states, attention_mask, center, hidden_states_for_route):
        expert_id = self.route(hidden_states, center,hidden_states_for_route)
        output = []
        for j in range(hidden_states.shape[0]):
            # print(attention_mask[j,:].shape)
            
            # expert_output = self.experts[expert_id[j].item()](hidden_states[j,:], attention_mask[j,:])
            
            expert_output = self.experts[expert_id[j].item()](hidden_states[j,:].view(1,self.config.seq_len, self.config.hidden_size), attention_mask[j,:].view(1,self.config.seq_len))
            output.append(expert_output)
        output = torch.cat(output, dim = 0)
        # print(output.shape)
        return output
