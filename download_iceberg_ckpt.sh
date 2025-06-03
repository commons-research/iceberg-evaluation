wget https://zenodo.org/record/8433354/files/canopus_iceberg_models.zip -O canopus_iceberg_models.zip
mkdir -p models/iceberg

mv canopus_iceberg_models.zip models/iceberg/canopus_iceberg_models.zip
unzip models/iceberg/canopus_iceberg_models.zip
mv canopus_iceberg_generate.ckpt models/iceberg/canopus_iceberg_generate.ckpt
mv canopus_iceberg_score.ckpt models/iceberg/canopus_iceberg_score.ckpt