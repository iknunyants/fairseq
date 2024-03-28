class TransformerFlops(object):
    def __init__(self, n_layers, n_heads, d_model, d_ff = None, d_heads = None):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_heads = d_heads if d_heads is not None else d_model / n_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        
    def linear_flops(self, in_features, out_features, len, density=1.0):
        return 2 * in_features * out_features * len * density
    
    def mm_flops(self, dim1, dim_both, dim2, density=1.0):
        return 2 * dim_both * dim1 * dim2 * density
    
    def layernorm_flops(self, len):
        return 5 * self.d_model * len
    
    def encoder_flops(self, src_len, density_dict):
        mm_flops, linear_flops, layernorm_flops = 0.0, 0.0, 0.0

        for i in range(self.n_layers):
            linear_flops += 3 * self.linear_flops(self.d_model, self.d_model, src_len, density_dict.get(f'enc_proj/layer_{i}', 1.0))
            
            mm_flops += self.mm_flops(src_len, self.d_heads, src_len, density_dict.get(f'enc_attention_q/layer_{i}', 1.0)) * self.n_heads

            mm_flops += self.mm_flops(src_len, src_len, self.d_heads, density_dict.get(f'enc_attention_v/layer_{i}', 1.0)) * self.n_heads
            
            linear_flops += self.linear_flops(self.d_model, self.d_model, src_len, density_dict.get(f'enc_attention_out/layer_{i}', 1.0))
            
            layernorm_flops += self.layernorm_flops(src_len)

            linear_flops += self.linear_flops(self.d_model, self.d_ff, src_len, density_dict.get(f'enc_fc1/layer_{i}', 1.0))

            linear_flops += self.linear_flops(self.d_ff, self.d_model, src_len, density_dict.get(f'enc_fc2/layer_{i}', 1.0))

            layernorm_flops += self.layernorm_flops(src_len)
        linear_flops += 2 * self.linear_flops(self.d_model, self.d_model, src_len, density_dict.get(f'encoder_out', 1.0))

        return [mm_flops, linear_flops, layernorm_flops]

    
    def decoder_flops(self, src_len, trg_len, density_dict):
        mm_flops, linear_flops, layernorm_flops = 0.0, 0.0, 0.0

        
        for i in range(self.n_layers):
            linear_flops += 3 * self.linear_flops(self.d_model, self.d_model, trg_len, density_dict.get(f'dec_proj/layer_{i}', 1.0))
            
            mm_flops += self.mm_flops(trg_len, self.d_heads, trg_len, density_dict.get(f'dec_attention_q/layer_{i}', 1.0)) * self.n_heads

            mm_flops += self.mm_flops(trg_len, trg_len, self.d_heads, density_dict.get(f'dec_attention_v/layer_{i}', 1.0)) * self.n_heads
            
            linear_flops += self.linear_flops(self.d_model, self.d_model, trg_len, density_dict.get(f'dec_attention_out/layer_{i}', 1.0))

            layernorm_flops += self.layernorm_flops(trg_len)

            linear_flops += self.linear_flops(self.d_model, self.d_model, trg_len, density_dict.get(f'dec_cross/layer_{i}', 1.0))

            mm_flops += self.mm_flops(trg_len, self.d_heads, src_len, density_dict.get(f'dec_crossattention_q/layer_{i}', 1.0)) * self.n_heads

            mm_flops += self.mm_flops(trg_len, src_len, self.d_heads, density_dict.get(f'dec_crossattention_v/layer_{i}', 1.0)) * self.n_heads

            linear_flops += self.linear_flops(self.d_model, self.d_model, trg_len, density_dict.get(f'dec_crossattention_out/layer_{i}', 1.0))

            layernorm_flops += self.layernorm_flops(trg_len)

            linear_flops += self.linear_flops(self.d_model, self.d_ff, trg_len, density_dict.get(f'dec_fc1/layer_{i}', 1.0))

            linear_flops += self.linear_flops(self.d_ff, self.d_model, trg_len, density_dict.get(f'dec_fc2/layer_{i}', 1.0))

            layernorm_flops += self.layernorm_flops(trg_len)


        return [mm_flops, linear_flops, layernorm_flops]

    def seq_flops(self, src_len, trg_len, density_dict={}, return_by_opp=False):
        dec_flops = [x / 1e9 for x in self.decoder_flops(src_len, trg_len, density_dict)]
        enc_flops = [x / 1e9 for x in self.encoder_flops(src_len, density_dict)]
        mm_flops, linear_flops, layernorm_flops = map(sum, zip(dec_flops, enc_flops))
        
        if return_by_opp:
            return [mm_flops, linear_flops, layernorm_flops]
        return sum([mm_flops, linear_flops, layernorm_flops])
    
    def batch_flops(self, src_len, trg_len, density_dict={}, bsz = 1, return_by_opp=False):
        if return_by_opp:
            return [x * bsz for x in self.seq_flops(src_len, trg_len, density_dict, return_by_opp)]
        return self.seq_flops(src_len, trg_len, density_dict, return_by_opp) * bsz
        