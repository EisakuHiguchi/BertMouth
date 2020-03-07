from transformers.modeling_bert import BertForMaskedLM, BertOnlyNSPHead
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

class BertMouth(BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    self.clsNSP = BertOnlyNSPHead(config)

    self.init_weights()

  def forward(
      self, input_ids=None, attention_mask=None, token_type_ids=None,
       position_ids=None, head_mask=None, inputs_embeds=None, 
       masked_lm_labels=None, encoder_hidden_states=None, 
       encoder_attention_mask=None, lm_labels=None,
       next_sentence_label=None, ):

    outputs = self.bert(
        input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
        position_ids=position_ids, head_mask=head_mask, 
        inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, 
        encoder_attention_mask=encoder_attention_mask, )
        
    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output)
    pooled_output = outputs[1]
    seq_relationship_score = self.clsNSP(pooled_output)

    outputs = (prediction_scores,) + outputs[2:]
    outputs = (seq_relationship_score,) + outputs

    if masked_lm_labels is not None:
      loss_fct = CrossEntropyLoss()
      masked_lm_loss = loss_fct(
          prediction_scores.view(-1, self.config.vocab_size), 
          masked_lm_labels.view(-1))
      outputs = (masked_lm_loss,) + outputs
    if lm_labels is not None:
      prediction_scores = prediction_scores[:, :-1, :].contiguous()
      lm_labels = lm_labels[:, 1:].contiguous()
      loss_fct = CrossEntropyLoss()
      ltr_lm_loss = loss_fct(
          prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
      outputs = (ltr_lm_loss,) + outputs

    # Next Sentence Prediction loss
    if next_sentence_label is not None:
      loss_fct = CrossEntropyLoss()
      next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
      outputs = (next_sentence_loss,) + outputs

    # classifier loss
    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)
    outputs = (logits, ) + outputs

    return outputs