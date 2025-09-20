import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.hidden_size = hidden_size

		self.input_weight = torch.rand(input_size,hidden_size, dtype=torch.float, requires_grad=True)
		self.output_weight = torch.rand(hidden_size, output_size, dtype=torch.float, requires_grad=True)
		self.rec_weight = torch.rand(hidden_size, hidden_size, dtype=torch.float, requires_grad=True)
		self.output_bias = torch.rand(output_size, dtype=torch.float, requires_grad=True)
		self.input_bias = torch.rand(hidden_size, dtype=torch.float, requires_grad=True)

		self.activation = nn.ReLU()

		# print(self.input_weight, self.output_weight, self.rec_weight, self.output_bias, self.input_bias, sep='\n')
		# print(self.input_weight.shape, self.output_weight.shape, self.rec_weight.shape, self.output_bias.shape, self.input_bias.shape, sep='\n')
		# print()

	def forward(self, x):	# Assuming batch dimension exists
		if x.ndim == 1:
			x = x.unsqueeze(0)

		previous = torch.zeros((x.shape[0], self.hidden_size))
		for timestep in range(0, x.shape[1]):
			cur_timestep = x[:, timestep].unsqueeze(1)
			w1b1 = cur_timestep @ self.input_weight + self.input_bias + previous @ self.rec_weight
			activated = self.activation(w1b1)
			previous = activated

			# print(cur_timestep, w1b1, activated, sep='\n')
			# print(cur_timestep.shape, w1b1.shape, activated.shape, sep='\n')
			# print()

		return previous @ self.output_weight + self.output_bias

	def get_params(self):
		return [self.input_weight, self.output_weight, self.rec_weight, self.output_bias, self.input_bias]


# Testing Part
model = RNN(1,4,2)

optimizer = Adam(model.get_params(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

criterion = MSELoss()
losses = []
for i in range(2000):
	answer = model(torch.tensor([[1,2,3],[1,5,4],[2,5,9],[9,4,7],[1,2,7]], dtype=torch.float))
	loss = criterion(answer, torch.tensor([[6,5],[2,3],[8,10],[8,9],[8,9]], dtype=torch.float))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	scheduler.step(loss.item())
	losses.append(loss.item())

final_answer = model(torch.tensor([[1,2,3],[1,5,4],[2,5,9],[9,4,7],[1,2,7]], dtype=torch.float))
print(final_answer, final_answer.shape)
plt.plot(losses)
plt.show()

