import torch
import tqdm


def calc_ppl(loss):
    """Calculate perplexity from Softmax Cross Entropy loss"""
    ppl = torch.exp(torch.tensor(loss)).item()
    return ppl


class Trainer:
    def __init__(self, model_output_dir, net, dataloader_dict, num_epochs, device, optimizer, scheduler, max_grad_norm, patience, writer=None, tqdm_disable=False):
        self._model_output_dir = model_output_dir
        self._net = net
        self._dataloader_dict = dataloader_dict
        self._num_epochs = num_epochs
        self._device = device
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._patience = patience
        self._tqdm_disable = tqdm_disable
        self._writer = writer

    def train(self):
        PHASE_TRAIN = "train"
        PHASE_VAL = "val"

        # keep the best model
        best = {
            "model": None,
            "epoch": 0,
            "loss": float("infinity"),
            "ppl": float("infinity"),
        }

        # 学習イテレーションの回数を保持
        num_iters = 0

        # keep the count which the validation metric does not improved
        num_patience = 0

        self._net.to(self._device)

        for epoch in range(self._num_epochs+1):
            print("Epoch {}/{}".format(epoch, self._num_epochs))
            # 学習と検証のループ
            for phase in [PHASE_TRAIN, PHASE_VAL]:
                # フェーズによってネットワークのモードを変更する
                # Dropout等の挙動に影響あり
                if phase == PHASE_TRAIN:
                    self._net.train()
                elif phase == PHASE_VAL:
                    self._net.eval()
                else:
                    raise Exception("got {} expected one of {}".format(phase, [PHASE_TRAIN, PHASE_VAL]))

                epoch_loss = 0

                # 未学習時の検証性能を確かめる
                if epoch == 0 and phase == PHASE_TRAIN:
                    continue

                for batch in tqdm.tqdm(self._dataloader_dict[phase], disable=self._tqdm_disable):
                    # GPUが使える場合はGPUにデータを送る
                    batch = {key: val.to(self._device) for key, val in batch.items()}

                    # Initialize optimizer
                    if phase == PHASE_TRAIN:
                        self._optimizer.zero_grad()

                    # set_grad_enabled(phrase=="train") で
                    # 学習時のみ勾配計算できるようにグラフ作成する
                    with torch.set_grad_enabled(phase==PHASE_TRAIN):
                        # labelsを指定することでlossを計算する
                        loss, _, _ = self._net(**batch)

                        if phase == PHASE_TRAIN:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self._net.parameters(), self._.max_grad_norm)
                            self._optimizer.step()
                            self._scheduler.step()

                            num_iters += 1

                        # epoch loss を更新
                        batch_size = batch["input_ids"].size()[0]
                        epoch_loss += loss.item() * batch_size

                        # TensorBoardへの描画を行う
                        # 学習時のみlossを描画
                        if phase == PHASE_TRAIN:
                            self._writer.add_scalars("train/loss", {phase: loss.item()}, num_iters)
                            self._writer.add_scalars("train/lr", {phase: self._scheduler.get_lr()[0]}, num_iters)

                epoch_loss = epoch_loss / len(self._dataloader_dict[phase].dataset)
                epoch_ppl = calc_ppl(epoch_loss)
                print("phase {}, loss: {:.4f}, ppl: {:.4f}".format(phase, epoch_loss, epoch_ppl))

                if self._writer and phase == PHASE_VAL:
                    self._writer.add_scalars("train/loss", {phase: epoch_loss}, num_iters)
                    self._writer.add_scalars("metric/ppl", {phase: epoch_ppl}, num_iters)

                    if best["loss"] > epoch_loss:
                        best = {"model": self._net, "epoch": epoch, "loss": epoch_loss, "ppl": epoch_ppl}
                        num_patience = 0
                        # save model
                        if self._model_output_dir:
                            print("Save model, epoch:", epoch)
                            self._net.save_pretrained(self._model_output_dir)
                    else:
                        num_patience += 1
                        print("Patience {}, epoch: {}".format(num_patience, epoch))

                    if num_patience > self._patience:
                        return
