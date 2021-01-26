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
      canvas: document.querySelector("canvas"),
    };
  },
  mounted() {
    const script = (p5) => {
      let a = tf.variable(tf.scalar(p5.random(-1, 1)));
      let b = tf.variable(tf.scalar(p5.random(-1, 1)));
      let c = tf.variable(tf.scalar(p5.random(-1, 1)));

      let x_vals = [];
      let y_vals = [];
      let dragging = false;
      let learningRate = 0.5;
      let optimizer;
      p5.setup = () => {
        p5.createCanvas(900, 600);
        p5.background(0);
        tf.tidy(() => (optimizer = tf.train.adam(learningRate)));
      };
      p5.draw = () => {
        //Point Draw İn Canvas
        p5.background(0);
        p5.stroke(255);
        p5.strokeWeight(4);
        for (let i = 0; i < x_vals.length; i++) {
          let x1 = p5.map(x_vals[i], -1, 1, 0, p5.width);
          let y1 = p5.map(y_vals[i], -1, 1, p5.height, 0);
          p5.point(x1, y1);
        }

        if (dragging) {
          let x = p5.map(p5.mouseX, 0, p5.width, -1, 1);
          let y = p5.map(p5.mouseY, 0, p5.height, 1, -1);
          x_vals.push(x);
          y_vals.push(y);
        } else {
          //Her çizimde modelimizi eğiteceğiz
          if (x_vals.length > 0) {
      
            tf.tidy(() => {
                      const ys = tf.tensor1d(y_vals);
              optimizer.minimize(() => p5.loss(p5.predict(x_vals), ys));
            });
          }
        }
        let curveX = [];
        for (let i = -1; i < 1.01; i += 0.05) {
          curveX.push(i);
        }
        tf.tidy(() => { 
          p5.beginShape();
          p5.stroke(255);
          p5.strokeWeight(2);
          p5.noFill();
          let ys = tf.tidy(() => p5.predict(curveX));
          let curveY = ys.dataSync();
          for (let i = 0; i < curveX.length; i++) {
            let x = p5.map(curveX[i], -1, 1, 0, p5.width);
            let y = p5.map(curveY[i], -1, 1, p5.height, 0);
            p5.vertex(x, y);
          }

          p5.endShape();


        });
        console.log(tf.memory().numTensors)
      };

      p5.mousePressed = () => {
        dragging = true;
      };
      p5.mouseReleased = () => {
        dragging = false;
      };

      p5.predict = (x) => {
        let xs = tf.tensor1d(x);
        // y=ax2+bx+c polinom fonksiyon denklemi
        return xs.square().mul(a).add(xs.mul(b)).add(c);
      };
      p5.loss = (pred, label) => pred.sub(label).square().mean();
    };
    //Attach
    this.canvas = new p5(script, this.canvas);
  },
};
</script>

<style>
</style>