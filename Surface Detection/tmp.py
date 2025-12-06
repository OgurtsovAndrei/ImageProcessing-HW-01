import kaggle




competition = 'vesuvius-challenge-surface-detection'
kaggle.api.competition_download_file(
                competition=competition,
                file_name="train_images/1061356924.tif",
                path="123",
                force=False,
                quiet=False
            )