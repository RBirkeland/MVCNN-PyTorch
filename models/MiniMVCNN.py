class MiniMVCNN(nn.Module):
    def __init__(self):
        super(MiniMVCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 122 * 122, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    # x = batch size x views x channel x width x height
    def forward(self, x):
        # Swap batch and views dims
        x = x.transpose(0, 1)
        #print(x.shape)

        # View pool
        view_pool = []

        for v in x:
            #print('1', v.shape)
            v = self.pool(F.relu(self.conv1(v)))
            #print('2', v.shape)
            v = self.pool(F.relu(self.conv2(v)))
            #print(v.shape)
            #plt.imshow(v[0][0].cpu().detach().numpy())
            #plt.show()
            v = v.view(-1, 16 * 122 * 122)
            #print('3', v.shape)

            # Add to array
            view_pool.append(v)

        # Pool views
        #pooled_view = torch.max(view_pool[0], view_pool[1])
        #pooled_view = torch.max(pooled_view, view_pool[2])

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])

        pooled_view = F.relu(self.fc1(pooled_view))
        # print('4', x.shape)
        pooled_view = F.relu(self.fc2(pooled_view))
        # print(x.shape)
        pooled_view = self.fc3(pooled_view)
        # print(x.shape)
        return pooled_view