<template>
  <body>
    <div id="app">
      
      <Poli/>
    </div>
  </body>
</template>

<script>
//import Canvas from "./components/Canvas.vue";
import Poli from "./components/Polinormal.vue";
export default {
  name: "App",
  components: {
  //  Canvas,
    Poli
  },
};
</script>

<style>
* {
  box-sizing: border-box;
}
body{
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
