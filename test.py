# for agent_id in range(self.n_agents):
#     self.critic_optimizer.zero_grad()
#     target_values = rewards_var[:, agent_id, :]
#     values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])
#     # print(target_values,1)
#     # print(type(target_values),type(values),values)
#     if self.critic_loss == "huber":
#         critic_loss = nn.functional.smooth_l1_loss(values, target_values)
#     else:
#         critic_loss = nn.MSELoss()(values, target_values)
#     critic_loss.backward()
#     if self.max_grad_norm is not None:
#         nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
#     self.critic_optimizer.step()
import torch

a = torch.tensor([0,1,4])
b = torch.tensor([2,5,6])
c = torch.tensor([7,0,49])

print(a+b)
d= a+b+c
print(d)
print(d/2)