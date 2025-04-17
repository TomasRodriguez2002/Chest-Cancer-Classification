import tensorflow as tf
import mlflow
import mlflow.tensorflow
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_trainer import Training
from src.cnnClassifier import logger

STAGE_NAME = "Training"

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, weights, include_top, num_classes, hp_params):
        self.input_shape = input_shape
        self.weights = weights
        self.include_top = include_top
        self.num_classes = num_classes
        self.hp_params = hp_params

    def build(self, hp):
        # Creamos VGG16 base fresh en cada trial
        base = tf.keras.applications.VGG16(
            input_shape=self.input_shape,
            weights=self.weights,
            include_top=self.include_top
        )

        # Congelamos segun tunable
        ft = self.hp_params["freeze_till"]
        freeze_to = hp.Int("freeze_till",
                           min_value=ft["min"],
                           max_value=ft["max"],
                           step=ft["step"])
        for layer in base.layers[:freeze_to]:
            layer.trainable = False

        x = tf.keras.layers.Flatten()(base.output)

        du = self.hp_params["dense_units"]
        units = hp.Int("dense_units",
                       min_value=du["min"],
                       max_value=du["max"],
                       step=du["step"])
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        out = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=base.input, outputs=out)

        lr = hp.Float("learning_rate",
                      min_value=self.hp_params["learning_rate"]["min"],
                      max_value=self.hp_params["learning_rate"]["max"],
                      sampling=self.hp_params["learning_rate"]["sampling"])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

class ModelTrainingPipeline:
    def main(self):
        cm = ConfigurationManager()
        tc = cm.get_training_config()

        training = Training(config=tc)
        training.train_valid_generator() # Generadores creados aquí

        # Calcular steps despues de generar los datos
        steps_per_epoch = training.train_generator.samples // training.train_generator.batch_size
        validation_steps = training.valid_generator.samples // training.valid_generator.batch_size

        train_gen = training.train_generator
        val_gen = training.valid_generator

        # MLflow autolog
        #mlflow.tensorflow.autolog()

        # Definimos tuner
        hypermodel = CNNHyperModel(
            input_shape=tc.params_image_size,
            weights=tc.params_weights,
            include_top=tc.params_include_top,
            num_classes=tc.params_classes,
            hp_params={
                "learning_rate": tc.hp_learning_rate,
                "freeze_till": tc.hp_freeze_till,
                "dense_units": tc.hp_dense_units
            }
        )

        tuner = RandomSearch(
            hypermodel,
            objective="val_accuracy",
            max_trials=tc.tuner_max_trials,
            executions_per_trial=tc.tuner_executions_per_trial,
            directory=str(tc.root_dir / "keras_tuner"),
            project_name="cnn_tuning"
        )

        with mlflow.start_run(run_name="keras_tuner_sweep"):
            tuner.search(
                train_gen,
                validation_data=val_gen,
                epochs=tc.params_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps
            )

            best_model = tuner.get_best_models(1)[0]
            best_hps = tuner.get_best_hyperparameters(1)[0]

            # Guardar modelo ganador
            best_model.save(tc.trained_model_path)
            mlflow.log_params(best_hps.values)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        ModelTrainingPipeline().main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise
