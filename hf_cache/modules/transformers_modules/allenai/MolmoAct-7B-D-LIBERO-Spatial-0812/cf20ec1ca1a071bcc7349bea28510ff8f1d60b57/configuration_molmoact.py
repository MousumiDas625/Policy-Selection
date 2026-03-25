"""
MolmoAct configuration
"""

from typing import Tuple, Optional, Dict, Any

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MolmoActVitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoActVisionTransformer`].
    It is used to instantiate a `MolmoActVisionTransformer` according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import MolmoActVitConfig, MolmoActVisionTransformer

    >>> # Initializing a MolmoActVitConfig
    >>> configuration = MolmoActVitConfig()

    >>> # Initializing a MolmoActVisionTransformer (with random weights)
    >>> model = MolmoActVisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmoact_vit"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        image_default_input_size: Tuple[int, int] = (378, 378),
        image_patch_size: int = 14,
        image_num_pos: int = 577,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        initializer_range: float = 0.02,
        float32_attention: bool = True,
        use_cls_token: bool = False,      # True for OpenCLIP
        patch_bias: bool = True,          # False for OpenCLIP
        pre_layernorm: bool = False,      # True for OpenCLIP
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.image_default_input_size = image_default_input_size
        self.image_patch_size = image_patch_size
        self.image_num_pos = image_num_pos
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.initializer_range = initializer_range
        self.float32_attention = float32_attention
        self.use_cls_token = use_cls_token
        self.patch_bias = patch_bias
        self.pre_layernorm = pre_layernorm

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


class MolmoActAdapterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of MolmoActAdapter. With MolmoActVitConfig,
    It is used to instantiate an MolmoActVisionBackbone according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import MolmoActVitConfig, MolmoActAdapterConfig, MolmoActVisionBackbone

    >>> # Initializing a MolmoActVitConfig and a MolmoActAdapterConfig
    >>> vit_config = MolmoActVitConfig()
    >>> adapter_config = MolmoPoolingConfig()

    >>> # Initializing a MolmoActVisionBackbone (with random weights)
    >>> model = MolmoActVisionBackbone(vit_config, adapter_config)

    >>> # Accessing the model configuration
    >>> vit_configuration = model.vit_config
    >>> adapter_configuration = model.adapter_config
    ```"""

    def __init__(
        self,
        vit_layers: Tuple = (-3, -9),
        hidden_size: int = 1152,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        hidden_act: str = "silu",
        intermediate_size: int = 18944,
        text_hidden_size: int = 3584,
        image_feature_dropout: float = 0.0,
        initializer_range: float = 0.02,
        # pooling_mode: str = "indices",            # "indices" (SigLIP) or "2x2_attention" (OpenCLIP)
        image_padding_embed: Optional[str] = None,  # e.g. "pad_and_partial_pad"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vit_layers = vit_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.text_hidden_size = text_hidden_size
        self.image_feature_dropout = image_feature_dropout
        self.initializer_range = initializer_range
        # self.pooling_mode = pooling_mode
        self.image_padding_embed = image_padding_embed


class MolmoActLlmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoActLlm`]. It is used to instantiate a
    `MolmoActLlm` according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import MolmoActLlmConfig, MolmoActLlm

    >>> # Initializing a MolmoActLlmConfig
    >>> configuration = MolmoActLlmConfig()

    >>> # Initializing a MolmoActLlm (with random weights)
    >>> model = MolmoActLlm(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmoact_llm"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "blocks.*.self_attn.att_proj": "colwise",
        "blocks.*.self_attn.attn_out": "rowwise",
        "blocks.*.mlp.ff_proj": "colwise",
        "blocks.*.mlp.ff_out": "rowwise",
    }
    base_model_pp_plan = {
        "wte": (["input_ids"], ["inputs_embeds"]),
        "blocks": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "ln_f": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: Optional[int] = 4,
        head_dim: int = 128,
        vocab_size: int = 152064,
        additional_vocab_size: int = 128,
        qkv_bias: bool = True,
        num_hidden_layers: int = 48,
        intermediate_size: int = 18944,
        hidden_act: str = "silu",
        embedding_dropout: float=0.0,
        attention_dropout: float=0.0,
        residual_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1000000.0,
        rope_scaling: Dict[str, Any] = None,
        use_qk_norm: bool = False,
        qk_norm_type: str = "olmo",
        layer_norm_eps: int = 1e-6,
        norm_after: bool = False,
        initializer_range: float = 0.02,
        use_cache=True,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.qkv_bias = qkv_bias
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.norm_after = norm_after
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)


class MolmoActConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MolmoActForActionReasoning`].
    It is used to instantiate an MolmoAct model according to the specified arguments, defining the model architecture.

    Example:

    ```python
    >>> from transformers import MolmoActConfig, MolmoActVitConfig, MolmoActAdapterConfig, MolmoActLlmConfig

    >>> # Initializing a MolmoActVitConfig
    >>> vit_config = MolmoActVitConfig()

    >>> # Initializing a MolmoActAdapterConfig
    >>> adapter_config = MolmoActAdapterConfig()

    >>> # Initializing a MolmoActLlmConfig
    >>> llm_config = MolmoActLlmConfig()

    >>> # Initializing a MolmoActConfig
    >>> configuration = MolmoActConfig(vit_config, adapter_config, llm_config, image_patch_id=152069)

    >>> # Initializing a model
    >>> model = MolmoActForActionReasoning(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmoact"
    sub_configs = {
        "llm_config": MolmoActLlmConfig,
        "vit_config": MolmoActVitConfig,
        "adapter_config": MolmoActAdapterConfig,
    }

    def __init__(
        self,
        vit_config: MolmoActVitConfig = None,
        adapter_config: MolmoActAdapterConfig = None,
        llm_config: MolmoActLlmConfig = None,
        image_patch_id: int = None,
        initializer_range: float = 0.02,
        n_action_bins: int = 256,
        norm_stats: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vit_config is None:
            self.vit_config = MolmoActVitConfig()
        elif isinstance(vit_config, dict):
            self.vit_config = MolmoActVitConfig(**vit_config)
        else:
            self.vit_config = vit_config
        if adapter_config is None:
            self.adapter_config = MolmoActAdapterConfig()
        elif isinstance(adapter_config, dict):
            self.adapter_config = MolmoActAdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config
        if llm_config is None:
            self.llm_config = MolmoActLlmConfig()
        elif isinstance(llm_config, dict):
            self.llm_config = MolmoActLlmConfig(**llm_config)
        else:
            self.llm_config = llm_config
        self.image_patch_id = image_patch_id
        self.initializer_range = initializer_range

        self.n_action_bins = n_action_bins
        self.norm_stats = norm_stats

    @property
    def image_num_patch(self):
        assert self.vit_config is not None
        return self.vit_config.image_num_patch
    
    @property
    def num_attention_heads(self):
        return self.llm_config.num_attention_heads
    
    @property
    def num_key_value_heads(self):
        return self.llm_config.num_key_value_heads

    @property
    def head_dim(self):
        return self.llm_config.head_dim

    @property
    def num_hidden_layers(self):
        return self.llm_config.num_hidden_layers
    
    @property
    def hidden_size(self):
        return self.llm_config.hidden_size
    
    @property
    def vocab_size(self):
        return self.llm_config.vocab_size
    
    @property
    def max_position_embeddings(self):
        return self.llm_config.max_position_embeddings


MolmoActVitConfig.register_for_auto_class()
MolmoActAdapterConfig.register_for_auto_class()
MolmoActLlmConfig.register_for_auto_class()
MolmoActConfig.register_for_auto_class()