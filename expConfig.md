
<table border="10">
<tr align="left"><th>No.</th>
<td>C1</td><td>C2</td><td>C3</td><td>C4</td><td>C5</td><td>C6</td><td>C7</td><td>C8</td><td>C9</td><td>C10</td>
</tr>

<tr align="left"><th>wav_mag_normalize</th>
<td>No</td><td>Yes</td><td></td><td>No</td><td>Yes</td><td>No</td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>log_spec_trunc</th>
<td>[-3,5]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>[-3,7]</td>
</tr>

<tr align="left"><th>input_size</th>
<td>257</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>output_size</th>
<td>257</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>rnn_size</th>
<td>496</td><td></td><td></td><td></td><td>1024</td><td>496</td><td></td><td></td><td>1024</td><td>496</td>
</tr>

<tr align="left"><th>rnn_layers_num</th>
<td>2</td><td></td><td>3</td><td>2</td><td>3</td><td>2</td><td></td><td>1</td><td>2</td><td></td>
</tr>

<tr align="left"><th>batch_size</th>
<td>128</td><td></td><td></td><td></td><td>64</td><td>256</td><td></td><td></td><td>128</td><td>256</td>
</tr>

<tr align="left"><th>learning_rate</th>
<td>0.001</td><td></td><td></td><td></td><td>0.002</td><td>0.001</td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>min_epoches</th>
<td>10</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>max_epoches</th>
<td>50</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>halving_factor</th>
<td>0.5</td><td></td><td></td><td></td><td></td><td>0.7</td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>start_halving_impr</th>
<td>0.003</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>end_halving_impr</th>
<td>0.001</td><td></td><td></td><td></td><td></td><td>0.0005</td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>keep_prob</th>
<td>0.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>max_grad_norm</th>
<td>5.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>model_type</th>
<td>LSTM</td><td></td><td></td><td>BLSTM</td><td>LSTM</td><td>BLSTM</td>
<td>LSTM</td><td>BLSTM</td><td></td><td></td>
</tr>

<tr align="left">
<th>final_train_loss</th><td>3.1817</td>
<td>4.1797</td><td>7.7716</td><td>=</td>
<td>7.5969</td><td>3.0427(0.9128)</td><td>3.8880(1.1664)</td><td>=</td><td>=</td><td>=</td>
</tr>

<tr align="left">
<th>final_validation_loss</th><td>5.0099</td>
<td>6.2518</td><td>7.9592</td><td>=</td><td>7.7481</td>
<td>3.0610(0.9183)</td><td>3.9230(1.1769)</td><td>=</td><td>=</td><td>=</td>
</tr>

<tr align="left">
<th>epoch_duration<br>(Normal)</th>
<td>5.5h</td><td>5.5h</td><td>7.0h</td><td>None</td><td>9.0h</td>
<td>None</td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left">
<th>epoch_duration<br>(SmallTFRecord+tfDataset)</th>
<td>3.0h</td><td>None</td><td></td><td></td><td></td>
<td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>epoch_duration<br>(BigTFRecord+tfDataset)</th>
<td>1.5h</td><td>None</td><td></td><td>=</td><td>None</td><td>1.8h</td>
<td>1.2h</td><td>=</td><td>5.9h</td><td>=</td>
</tr>

<tr align="left"><th>epoch_duration<br>(BigTFRecord+tfQueue)</th>
<td>1.9h</td><td>None</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>speaker num</th>
<td>4</td><td></td><td></td><td></td><td></td><td>90</td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>train_set speech duration</th><td>1126h</td><td></td><td></td><td></td><td></td><td>1166h</td>
<td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th>utterance duration</th>
<td>10S</td><td></td><td></td><td></td><td></td><td>3S</td><td></td><td></td><td></td><td></td>
</tr>

<tr align="left"><th colspan='11'>表中留空位置的值表示该参数与其前面的实验参数相同，“=”表示实验尚未得出结果或未进行该实验。</th>
</tr>
</table>
