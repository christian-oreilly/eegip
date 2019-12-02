# EEG IP package

In developement. Not meant for reuse by external party yet. Project in constant flux!

# Campaings
Campaign have four levels, and for each levels, there is 
a set of properties that can be modified in a factorial design. These
levels are as follow:
 - Preprocessing:
     - dataset: 
        - EEGIP current datasets: ['london06', 'london12', 'washington']
 - Source reconstruction:
     - con_type: ['scalp', 'sources']
     - inv_method: 
        - MNE current inv_method: ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
     - event_type: Depends on the dataset. See the YAML configuration file.
 - Connectivity computation:
     - method: 
        - MNE current method: [ 'coh', 'cohy', 'imcoh', 'plv', 'ppc', 'pli', 'pli2_unbiased', 'wpli', 'wpli2_debiased']
     - band: User defined, in the YAM configuration file.
 - Connectivity aggregation

 
 
# File
 
 We use the following file nomenclature:
 - EEGIP recordings: 
   - pattern: [DATA][PATTERN]qcr.set
   - example: /project/def-emayada/eegip/london/derivatives/lossless/sub-s601/ses-m06/eeg/sub-s601_ses-m06_eeg_qcr.set
 - Preprocessed recordings: 
   - pattern: [OUT][PATTERN]qcr-raw.fif
   - example: /project/def-emayada/oreillyc/eegip/london/sub-s601/ses-m06/eeg/sub-s601_ses-m06_eeg_qcr-raw.fif
 - Epoch:
   - pattern: [OUT][PATTERN]qcr-epo.fif
   - example: /project/def-emayada/oreillyc/eegip/london/sub-s601/ses-m06/eeg/sub-s601_ses-m06_eeg_qcr-epo.fif
 - Evoked:
   - pattern: [OUT][PATTERN]qcr-ave.fif
   - example: /project/def-emayada/oreillyc/eegip/london/sub-s601/ses-m06/eeg/sub-s601_ses-m06_eeg_qcr-ave.fif
 - Sources:
   - pattern: [OUT][PATTERN]qcr-{event_type}-{key1=val1}-...-{keyN=valN}-{con_type}.npy
   - example: /project/def-emayada/oreillyc/eegip/london/sub-s601/ses-m06/eeg/sub-s601_ses-m06_eeg_qcr-direct-dSPM-sources.npy
 - Labels:
   - pattern: [OUT][PATTERN]qcr-{event_type}-{key1=val1}-...-{keyN=valN}-sources-labels.npy
   - example: /project/def-emayada/oreillyc/eegip/london/sub-s601/ses-m06/eeg/sub-s601_ses-m06_eeg_qcr-direct-dSPM-sources-labels.npy
 - Connectivity matrix: 
    - pattern: [OUT][PATTERN]qcr-{fmin}-{fmax}-{event_type}-{key1=val1}-...-{keyN=valN}-{con_type}-con.csv
   - example: /project/def-emayada/oreillyc/eegip/london/sub-s601/ses-m06/eeg/sub-s601_ses-m06_eeg_qcr-100-200-direct-wpli-scalp-con.csv 
 - Connectivity aggregates: 
    - pattern: [OUT]/con_matrix-{con_type}-{key1=val1}-...-{keyN=valN}.pck
    - example: /project/def-emayada/oreillyc/eegip/london06/con_matrix_scalp_wpli.pck 
 - BEM model: 
    - pattern: [ATLAS]/...
 - Atlas parcel meshes Meshes
    - pattern: [ATLAS]/...
 - Demo files
    - pattern: [DEMO]/...
 - Slurm log files
    - pattern: [OUT_ROOT]/log/...
 - Sbatch scripts
    - pattern: [OUT_ROOT]/slurm/...
     
     
   
 

     





