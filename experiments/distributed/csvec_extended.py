import torch

try:
    from csh.csvec import CSVec
except:
    # print("Import csh by running: $git submodule update --init")
    raise Exception("In order to execute the count sketch simulations, import csh by running *in main project folder*:\n"
                    "$ git clone https://github.com/nikitaivkin/csh")

class CSVec_Extended(CSVec):
    def to_1d_tensor(self):
        return self.table.view(-1)

    def from_1d_tensor(self, tbl):
        self.table = torch.reshape(tbl, self.table.shape)
