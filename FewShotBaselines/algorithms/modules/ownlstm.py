from __future__ import division, print_function, absolute_import

import pdb
import math
import torch
import torch.nn as nn

class MetaLSTMCell(nn.Module):
    """
    The final (meta-learner) LSTM cell that is stacked on top of a regular LSTM
    
    ...

    Attributes
    ----------
    input_size : int
        First layer of the LSTM meta-learner 
    hidden_size : int
        Number of nodes in the hidden layer of the first LSTM cell
    n_learner_params : int
        Number of parameters in the base-learner
    WF : nn.Parameter
        Weight matrix of the forget gate
    WI : nn.Parameter
        Weight matrix for the learning rate
    cI : nn.Parameter
        Cell state which contains base-learner parameters (initialization)
    bI : nn.Parameter
        Bias for the learning rate parameterization
    bF : nn.Parameter
        Bias for the forget gate
        
    Methods
    -------
    reset_parameters()
        Resets the parameters of the MetaLSTM
    init_cI()
        Initialize the cell state of the model
    forward(inputs, hx=None)
        Unroll the LSTM for a single step
    extra_repr()
        Return partial state string of the meta LSTM cell
    """
    
    def __init__(self, input_size, hidden_size, n_learner_params):
        """
        Initialize all attributes
        
        Parameters
        ----------
        input_size : int
            First layer of the LSTM meta-learner 
        hidden_size : int
            Number of nodes in the hidden layer of the first LSTM cell
            SHOULD BE 1
        n_learner_params : int
            Number of parameters in the base-learner
        """
        
        super(MetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_learner_params = n_learner_params
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.cI = nn.Parameter(torch.Tensor(n_learner_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all parameters 
        
        Resets the parameters of the MetaLSTM in a way that
        the forget gate will be large and input values low 
        such that the model starts with gradient descent
        """
        
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        nn.init.uniform_(self.bF, 4, 6)
        nn.init.uniform_(self.bI, -5, -4)

    def init_cI(self, flat_params):
        """Initialize the cell state
        
        Initializes the cell state with the provided base-learner parameters
        
        Parameters
        ----------
        flat_params : torch.Tensor
            Flattened base-learner parameters 
        """
        
        self.cI.data.copy_(flat_params.unsqueeze(1))

    def forward(self, inputs, hx=None):
        """Unroll the meta LSTM network for a single step
        
        Take the provided inputs and previous state information (hx) to produce
        a single update of the base-learner parameters by unrolling the meta LSTM
        for a single step
        
        Parameters
        ----------
        inputs : list
            Output of the first layer LSTM and gradients: [lstmhx, grad]
            inputs = [x_all, grad]:
                x_all (torch.Tensor of size [n_learner_params, input_size]): outputs from previous LSTM
                grad (torch.Tensor of size [n_learner_params]): gradients from learner
        hx : torch.Tensor
            Previous hidden state of the meta LSTM cell
            format = [f_prev, i_prev, c_prev]:
                f (torch.Tensor of size [n_learner_params, 1]): forget gate
                i (torch.Tensor of size [n_learner_params, 1]): input gate
                c (torch.Tensor of size [n_learner_params, 1]): flattened learner parameters
        
        Returns
        ----------
        c_next
            Next cell state
        f_next, i_next, c_next
            Next forget gate, learning rate, and cell state
        
        """
        
        """C_t = f_t * C_{t-1} + i_t * \tilde{C_t}"""

        x_all, grad = inputs
        batch, _ = x_all.size()

        if hx is None:
            f_prev = torch.zeros((batch, self.hidden_size)).to(self.WF.device)
            i_prev = torch.zeros((batch, self.hidden_size)).to(self.WI.device)
            c_prev = self.cI
            hx = [f_prev, i_prev, c_prev]

        f_prev, i_prev, c_prev = hx
        
        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), 1), self.WF) + self.bF.expand_as(f_prev)
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), 1), self.WI) + self.bI.expand_as(i_prev)
        # next cell/params
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)

        return c_next, [f_next, i_next, c_next]

    def extra_repr(self):
        """Return string representation of state
        
        Produce a string of format: input_size, hidden_size, n_learner_params
        
        Returns
        ----------
        string
            State of the Meta LSTM cell of the form: 
            input_size, hidden_size, n_learner_params
        """
        
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)


class MetaLearner(nn.Module):
    """
    LSTM meta learner module
    
    ...

    Attributes
    ----------
    lstm : nn.LSTMCell
        First layer of the LSTM meta-learner 
    metalstm : MetaLSTMCell
        Final layer of the LSTM meta-learner which embeds the base-learner weights
        
    Methods
    -------
    forward(inputs, hs=None)
        Perform a single unroll step on the given inputs
    """

    def __init__(self, input_size, hidden_size, n_learner_params):
        """
        Parameters
        ----------
        input_size : int
            Dimensionality of the input data (number of columns in the input matrix)
        hidden_size : int
            Number of hidden nodes for the first LSTM layer
        n_learner_params : int
            Number of parameters in the base-learner
        """
        
        super(MetaLearner, self).__init__()
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)

    def forward(self, inputs, hs=None):
        """Perform a single unroll step of the Meta LSTM
        
        Takes the inputs and performs an update of the base-learner parameters
        
        Parameters
        ----------
        inputs : list
            Input to the LSTM meta-learner in the format [loss, grad_prep, grad]
             loss : torch.Tensor of size (1,2) -- (sign, value)
             grad_prep : torch.Tensor of size (n_learner_params, 2)
             grad : torch.Tensor of size n_learner_params
        hs : list, optional
            Information from previous hidden states, in format 
            [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
            (default=None)
        """
        
        loss, grad_prep, grad = inputs
        loss = loss.expand_as(grad_prep)
        inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4]

        if hs is None:
            hs = [None, None]

        lstmhx, lstmcx = self.lstm(inputs, hs[0])
        flat_learner_unsqzd, metalstm_hs = self.metalstm([lstmhx, grad], hs[1])

        return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]
    
    

