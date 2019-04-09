class ConstantConfig:
    # ----- CONSTANTS -----
    # Random seed
    seed = 1234

    # Logging level
    level = 'DEBUG'

    # Out-of-vocabulary token string
    oov = '<unk>'

    # These are end-of-sequence marks
    end_sym_utterance = '</s>'

    # Special tokens need to be defined here, because model architecture may adapt depending on these
    unk_sym = 0  # Unknown word token <unk>
    eos_sym = 1  # end-of-utterance symbol </s>
    eod_sym = 2  # end-of-dialogue symbol </d>
    first_speaker_sym = 3  # first speaker symbol <first_speaker>
    second_speaker_sym = 4  # second speaker symbol <second_speaker>
    third_speaker_sym = 5  # third speaker symbol <third_speaker>
    minor_speaker_sym = 6  # minor speaker symbol <minor_speaker>
    voice_over_sym = 7  # voice over symbol <voice_over>
    off_screen_sym = 8  # off screen symbol <off_screen>
    pause_sym = 9  # pause symbol <pause>


class ModelArchConfig:
    # ----- MODEL ARCHITECTURE -----
    # If this flag is on, the hidden state between RNNs in subsequences is always initialized to zero.
    # Set this to reset all RNN hidden states between 'max_grad_steps' time steps
    reset_hidden_states_between_subsequences = False

    # If this flag is on, the maxout activation function will be applied to the utterance decoders output unit.
    # This requires qdim_decoder = 2x rankdim
    maxout_out = False

    # If this flag is on, a two-layer MLPs will applied on the utterance decoder hidden state before 
    # outputting the distribution over words.
    deep_out = True

    # If this flag is on, there will be an extra MLP between utterance and dialogue encoder
    deep_dialogue_input = False

    # Default and recommended setting is: tanh.
    # The utterance encoder and utterance decoder activation function
    sent_rec_activation = 'lambda x: T.tanh(x)'

    # The dialogue encoder activation function
    dialogue_rec_activation = 'lambda x: T.tanh(x)'

    # Determines how to input the utterance encoder and dialogue encoder into the utterance decoder RNN hidden state:
    #  - 'first': initializes first hidden state of decoder using encoders
    #  - 'all': initializes first hidden state of decoder using encoders, 
    #            and inputs all hidden states of decoder using encoders
    #  - 'selective': initializes first hidden state of decoder using encoders, 
    #                 and inputs all hidden states of decoder using encoders.
    #                 Furthermore, a gating function is applied to the encoder input 
    #                 to turn off certain dimensions if necessary.
    #
    # Experiments show that 'all' is most effective.
    decoder_bias_type = 'all'

    # Define the gating function for the three RNNs.
    utterance_encoder_gating = 'GRU'  # Supports 'None' and 'GRU'
    dialogue_encoder_gating = 'GRU'  # Supports 'None' and 'GRU'
    utterance_decoder_gating = 'GRU'  # Supports 'None', 'GRU' and 'LSTM'

    # If this flag is on, two utterances encoders (one forward and one backward) will be used,
    # otherwise only a forward utterance encoder is used.
    bidirectional_utterance_encoder = False

    # If this flag is on, there will be a direct connection between utterance encoder and utterance decoder RNNs.
    direct_connection_between_encoders_and_decoder = False

    # If this flag is on, there will be an extra MLP between utterance encoder and utterance decoder.
    deep_direct_connection = False

    # If this flag is on, the model will collapse to a standard RNN:
    # 1) The utterance+dialogue encoder input to the utterance decoder will be zero
    # 2) The utterance decoder will never be reset
    # Note this model will always be initialized with a hidden state equal to zero.
    collapse_to_standard_rnn = False

    # If this flag is on, the utterance decoder will be reset after each end-of-utterance token.
    reset_utterance_decoder_at_end_of_utterance = True

    # If this flag is on, the utterance encoder will be reset after each end-of-utterance token.
    reset_utterance_encoder_at_end_of_utterance = True


class HiddenLayerConfig:
    # ----- HIDDEN LAYER DIMENSIONS -----
    # Dimensionality of (word-level) utterance encoder hidden state
    qdim_encoder = 512

    # Dimensionality of (word-level) utterance decoder (RNN which generates output) hidden state
    qdim_decoder = 512

    # Dimensionality of (utterance-level) context encoder hidden layer
    sdim = 1000

    # Dimensionality of low-rank word embedding approximation
    rankdim = 256


class LatentVariableConfig:
    # ----- LATENT VARIABLES WITH VARIATIONAL LEARNING -----
    # If this flag is on, a Gaussian latent variable is added at the beginning of each utterance.
    # The utterance decoder will be conditioned on this latent variable,
    # and the model will be trained using the variational lower bound. 
    # See, for example, the variational auto-encoder by Kingma et al. (2013).
    add_latent_gaussian_per_utterance = False

    # This flag will condition the latent variable on the dialogue encoder
    condition_latent_variable_on_dialogue_encoder = False

    # This flag will condition the latent variable on the DCGM (mean pooling over words) encoder.
    # This will replace the conditioning on the utterance encoder.
    # If the flag is false, the latent variable will be conditioned on the utterance encoder RNN.
    condition_latent_variable_on_dcgm_encoder = False

    # Dimensionality of Gaussian latent variable, which has diagonal covariance matrix.
    latent_gaussian_per_utterance_dim = 10

    # If this flag is on, the latent Gaussian variable at time t will be affected linearly
    # by the distribution (sufficient statistics) of the latent variable at time t-1.
    # This is different from an actual linear state space model (Kalman filter),
    # since effective latent variables at time t are independent of all other latent variables,
    # given the observed utterances. However, it's useful, because it avoids forward propagating noise
    # which would make the training procedure more difficult than it already is.
    # Although it has nice properties (matrix preserves more information since it is full rank,
    # and if its eigenvalues are all positive the linear dynamics are just rotations in space),
    # it appears to make training very unstable!
    latent_gaussian_linear_dynamics = False

    # This is a constant by which the diagonal covariance matrix is scaled.
    # By setting it to a high number (e.g. 1 or 10),
    # the KL divergence will be relatively low at the beginning of training.
    scale_latent_variable_variances = 10

    # If this flag is on, the utterance decoder will ONLY be conditioned on the Gaussian latent variable.
    condition_decoder_only_on_latent_variable = False

    # If this flag is on, the KL-divergence term weight for the Gaussian latent variable
    # will be slowly increased from zero to one.
    train_latent_gaussians_with_kl_divergence_annealing = False

    # The KL-divergence term weight is increased by this parameter for every training batch.
    # It is truncated to one. For example, 1.0/60000.0 means that at iteration 60000 the model
    # will assign weight one to the KL-divergence term
    # and thus only be maximizing the true variational bound from iteration 60000 and onward.
    kl_divergence_annealing_rate = 1.0 / 60000.0

    # If this flag is enabled, previous token input to the decoder RNN is replaced with 'unk' tokens at random.
    decoder_drop_previous_input_tokens = False

    # The rate at which the previous tokens input to the decoder is kept (not set to 'unk').
    # Setting this to zero effectively disables teacher-forcing in the model.
    decoder_drop_previous_input_tokens_rate = 0.75

    # Initialization configuration
    initialize_from_pretrained_word_embeddings = False
    fix_pretrained_word_embeddings = False

    # If this flag is on, the model will fix the parameters of the utterance encoder and dialogue encoder RNNs,
    # as well as the word embeddings. NOTE: NOT APPLICABLE when the flag 'collapse_to_standard_rnn' is on.
    fix_encoder_parameters = False


class TrainingConfig:
    # ----- TRAINING PROCEDURE -----
    # Choose optimization algorithm (adam works well most of the time)
    updater = 'adam'

    # If this flag is on, NCE (Noise-Contrastive Estimation) will be used to train model.
    # This is significantly faster for large vocabularies (e.g. more than 20K words), 
    # but experiments show that this degrades performance.
    use_nce = False

    # Threshold to clip the gradient
    cutoff = 1.

    # Learning rate. The rate 0.0002 seems to work well across many tasks with adam.
    # Alternatively, the learning rate can be adjusted down (e.g. 0.00004) 
    # to at the end of training to help the model converge well.
    lr = 0.0002

    # Early stopping configuration
    patience = 20
    cost_threshold = 1.003

    # Batch size. If out of memory, modify this!
    bs = 20

    # Sort by length groups of
    sort_k_batches = 20

    # Training examples will be split into subsequences.
    # This parameter controls the maximum size of each subsequence.
    # Gradients will be computed on the subsequence, and the last hidden state of all RNNs will
    # be used to initialize the hidden state of the RNNs in the next subsequence.
    max_grad_steps = 80

    # Modify this in the prototype
    save_dir = None

    # Frequency of training error reports (in number of batches)
    train_freq = 10

    # Validation frequency
    valid_freq = 5000

    # Number of batches to process
    loop_iters = 3000000

    # Maximum number of minutes to run
    # This is one month.
    time_stop = 24 * 60 * 31

    # Error level to stop at
    minerr = -1


class DatasetConfig:
    train_dialogues = None
    test_dialogues = None
    valid_dialogues = None
    dictionary = None
    pretrained_word_embeddings_file = None


class BasicConfig(ConstantConfig,
                  ModelArchConfig,
                  HiddenLayerConfig,
                  LatentVariableConfig,
                  TrainingConfig):
    pass
