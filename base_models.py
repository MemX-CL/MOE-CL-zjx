import torch.nn as nn
import torch
from transformer.Transformer import BertModel, BertDecoderModel, BertLayerSaveModel, Embeddings, new_expert, new_layer, simple_layer, TransformerEncoder,vermilion_layer, rose_layer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead,BertPooler
from Transformer_MOE import BertModelWithMOE

class BertForMLM(nn.Module):
    def __init__(self, config):
        super(BertForMLM, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores
class BertForMLM_tawny(nn.Module):
    def __init__(self, config):
        super(BertForMLM_tawny, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, output
    
    
class BertWithDecoders(nn.Module):
    def __init__(self, config):
        super(BertWithDecoders, self).__init__()
        self.config = config
        self.bert = BertDecoderModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask)
        
        scores = self.head(outputs[0])
        outputs[0] = scores
        
        # replicated_labels = [labels for _ in range(len(outputs))]
        # losses = [self.criterion(output.view(-1, self.config.vocab_size), target.view(-1)) for output, target in zip(outputs, replicated_labels)]
        return outputs
    
    
class BertWithSavers(nn.Module):
    def __init__(self, config):
        super(BertWithSavers, self).__init__()
        self.config = config
        self.bert = BertLayerSaveModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        output, layer_outputs, ffn_outputs = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, layer_outputs, ffn_outputs
    


class BertWithSaversX(nn.Module):
    def __init__(self, config):
        super(BertWithSaversX, self).__init__()
        self.config = config
        self.bert = BertLayerSaveModel(config)
        self.head = BertOnlyMLMHead(config)
        self.pooler = BertPooler(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        output, layer_outputs = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, layer_outputs


class new_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([new_layer(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 42
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            inputs[i][expert_id].append(hidden_states)
            # print(expert_id)
            hidden_states = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            outputs[i][expert_id].append(hidden_states)
            expert_ids.append(expert_id)
            
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids


class vermilion_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([vermilion_layer(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 42
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states = self.layers[i](hidden_states, attention_mask, cluster_centers[i])

            expert_ids.append(expert_id)
            
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids



class simple_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ahead_num = 3
        cluster_num = config.num_experts
        unique_dim = 160
        heads = 12
        self.layers1 = nn.ModuleList([TransformerEncoder(config) for i in range(ahead_num)])
        self.special_layer = simple_layer(self.config, cluster_num, unique_dim, 8)
        self.layers2 = nn.ModuleList([TransformerEncoder(config) for i in range(ahead_num+1, config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 42
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers, pca):
        hidden_states = self.embeddings(input_ids)
        outputs = []
        for i in range(len(self.layers1)):
            hidden_states = self.layers1[i](hidden_states, attention_mask)
            outputs.append(hidden_states)
        hidden_states = self.special_layer(hidden_states, attention_mask, cluster_centers, pca)
        outputs.append(hidden_states)
        for j in range(len(self.layers2)):
            hidden_states = self.layers2[j](hidden_states, attention_mask)
            outputs.append(hidden_states)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, outputs

class BertWithMOE0(nn.Module):
    def __init__(self, config):
        super(BertWithMOE0, self).__init__()
        self.config = config
        self.bert = BertModelWithMOE(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, routes = self.bert(input_ids, attention_mask)
        raw_output = output
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, routes
class BertWithMOE(nn.Module):
    def __init__(self, config):
        super(BertWithMOE, self).__init__()
        self.config = config
        self.bert = BertModelWithMOE(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, routes = self.bert(input_ids, attention_mask)
        raw_output = output
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, routes, raw_output
    


class rose_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([rose_layer(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 42
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers, hidden_states_for_router):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i], hidden_states_for_router[i])
            # print(expert_id)
            hidden_states = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hidden_states_for_router[i] )

            expert_ids.append(expert_id)
            
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids