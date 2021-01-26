<template>
  <div class="canvas-container">
    <div class="canvas"></div>
  </div>
</template>

<script>
import p5 from "p5";
import * as tf from "@tensorflow/tfjs";
export default {
  data() {
    return {
      p5: "",
      canvas: document.querySelector(".canvas"),
    };
  },
  mounted() {
    let m;
    let b;
    const script = (p5) => {
      // These are your typical setup() and draw() methods

      this.p5 = p5;
      let x_vals = [];
      let y_vals = [];
      const learningRate = 0.01;
      let optimizer;
      //Modelimizin hata payını minimize ediyoruz
      tf.tidy(() => {
        optimizer = tf.train.sgd(learningRate);
      });
      p5.setup = () => {
        p5.createCanvas(900, 600);
        p5.background(0);

        //Sonradan Üzerinde Değişiklik Yapılabilen Tensorler
        m = tf.variable(tf.scalar(p5.random(1)));
        b = tf.variable(tf.scalar(p5.random(1)));
      };
      p5.draw = () => {
        p5.background(0); //her seferde clean bg
        p5.stroke(255);
        p5.strokeWeight(4);

        //Hata Değerlerini Minimize Ediyoruz


        //Train the model
        if (x_vals.length > 0) {
          //Bllek Optimizesi için
          tf.tidy(() => {
            let ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => p5.loss(p5.predict(x_vals), ys));
          });
        }
        for (let i = 0; i < x_vals.length; i++) {
          let px = p5.map(x_vals[i], 0, 1, 0, p5.width);
          let py = p5.map(y_vals[i], 0, 1, p5.height, 0);
          p5.point(px, py);
        }

        tf.tidy(() => {
          let lineX = [0, 1];
          const ys = p5.predict(lineX);
          let lineY = ys.dataSync();

          let x1 = p5.map(lineX[0], 0, 1, 0, p5.width);
          let x2 = p5.map(lineX[1], 0, 1, 0, p5.width);

          let y1 = p5.map(lineY[0], 0, 1, p5.height, 0);
          let y2 = p5.map(lineY[1], 0, 1, p5.height, 0);
          //Sağlanan nesnede bulunan tüm tf.Tensor ları ortadan kaldırır .
          ys.dispose();
          p5.strokeWeight(2);
          p5.line(x1, y1, x2, y2);
        });

        console.log(tf.memory().numTensors);
      };
      p5.mousePressed = () => {
        let x = p5.map(p5.mouseX, 0, p5.width, 0, 1);
        let y = p5.map(p5.mouseY, 0, p5.height, 1, 0);
        x_vals.push(x);
        y_vals.push(y);
      };

      p5.loss = (pred, label) => pred.sub(label).square().mean();

      //y değerini return eder
      p5.predict = (x) => {
        const xs = tf.tensor1d(x);
        //y=mx+b Doğru Formülü
        return xs.mul(m).add(b);
      };
    };

    // Attach the canvas to the div
    this.canvas = new p5(script, this.canvas);
  },
};
</script>

<style scoped>
.canvas-container {
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #f4f4f4;
}
.canvas {
  margin-top: 8rem;
}
</style>