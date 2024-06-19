import torch 

class PatientsInfosDataset(torch.utils.data.Dataset):
  def __init__(self, patients_infos, auxiliary_patients_infos, labels):
      self.patients_basic_info = torch.Tensor(patients_infos)
      self.patients_auxiliary_infos = torch.Tensor(auxiliary_patients_infos)
      self.labels = torch.Tensor(labels)

  def __len__(self):
      return len(self.patients_basic_info)

  def __getitem__(self, index):

      patients_basic_info = self.patients_basic_info[index]
      patients_auxiliary_infos = self.patients_auxiliary_infos[index]
      labels = self.labels[index]
      return patients_basic_info, patients_auxiliary_infos, labels
