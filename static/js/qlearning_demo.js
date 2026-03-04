(() => {
  // Gridworld: 5x5, start top-left, goal bottom-right, a few obstacles
  const N = 5;
  const start = [0, 0];
  const goal = [4, 4];
  const obstacles = new Set(["1,1", "2,1", "3,1", "3,3"]); // tweak as you like

  // Rewards
  const R_STEP = -1;
  const R_GOAL = 10;

  // Q-learning hyperparams
  let alpha = 0.2;
  let gamma = 0.95;
  let epsilon = 0.90;
  const epsMin = 0.05;
  const epsDecay = 0.985;

  // State
  let episode = 0;
  let agent = [...start];

  // Q-table: key "r,c" -> [up, right, down, left]
  const Q = new Map();
  const actions = [
    [-1, 0], // up
    [0, 1],  // right
    [1, 0],  // down
    [0, -1]  // left
  ];

  // DOM
  const gridEl = document.getElementById("grid");
  const kEpisode = document.getElementById("kpiEpisode");
  const kEps = document.getElementById("kpiEps");
  const kSteps = document.getElementById("kpiSteps");
  const kReward = document.getElementById("kpiReward");

  const btnTrain100 = document.getElementById("train100");
  const btnRun1 = document.getElementById("run1");
  const btnReset = document.getElementById("reset");

  function keyOf(rc) { return `${rc[0]},${rc[1]}`; }
  function inBounds(r, c) { return r >= 0 && r < N && c >= 0 && c < N; }
  function isObstacle(r, c) { return obstacles.has(`${r},${c}`); }
  function isGoal(r, c) { return r === goal[0] && c === goal[1]; }

  function ensureQ(stateKey){
    if (!Q.has(stateKey)) Q.set(stateKey, [0,0,0,0]);
    return Q.get(stateKey);
  }

  function argmax(arr){
    let bestI = 0, bestV = arr[0];
    for (let i=1;i<arr.length;i++){
      if (arr[i] > bestV){
        bestV = arr[i]; bestI = i;
      }
    }
    return bestI;
  }

  function pickAction(stateKey){
    const q = ensureQ(stateKey);
    if (Math.random() < epsilon){
      // explore: random valid move
      const valid = [];
      for (let a=0; a<4; a++){
        const nr = agent[0] + actions[a][0];
        const nc = agent[1] + actions[a][1];
        if (inBounds(nr,nc) && !isObstacle(nr,nc)) valid.push(a);
      }
      return valid[Math.floor(Math.random()*valid.length)];
    }
    // exploit
    return argmax(q);
  }

  function step(actionIdx){
    const [dr, dc] = actions[actionIdx];
    const nr = agent[0] + dr;
    const nc = agent[1] + dc;

    // If invalid, stay put with step penalty
    if (!inBounds(nr,nc) || isObstacle(nr,nc)){
      return { next: [...agent], reward: R_STEP };
    }
    const reward = isGoal(nr,nc) ? R_GOAL : R_STEP;
    return { next: [nr,nc], reward };
  }

  function trainEpisode(maxSteps=80){
    let totalReward = 0;
    let steps = 0;
    agent = [...start];

    while (steps < maxSteps){
      const sKey = keyOf(agent);
      const a = pickAction(sKey);
      const { next, reward } = step(a);

      const nsKey = keyOf(next);
      const qS = ensureQ(sKey);
      const qNS = ensureQ(nsKey);

      // Q update
      const bestNext = Math.max(...qNS);
      qS[a] = qS[a] + alpha * (reward + gamma * bestNext - qS[a]);

      agent = next;
      totalReward += reward;
      steps++;

      if (isGoal(agent[0], agent[1])) break;
    }

    episode++;
    epsilon = Math.max(epsMin, epsilon * epsDecay);

    return { steps, totalReward };
  }

  function render(){
    gridEl.innerHTML = "";

    for (let r=0;r<N;r++){
      for (let c=0;c<N;c++){
        const cell = document.createElement("div");
        cell.className = "cell glass";

        if (isObstacle(r,c)) cell.classList.add("obstacle");
        if (r === start[0] && c === start[1]) cell.classList.add("start");
        if (r === goal[0] && c === goal[1]) cell.classList.add("goal");

        if (r === agent[0] && c === agent[1]){
          const dot = document.createElement("div");
          dot.className = "agent";
          cell.appendChild(dot);
        }

        gridEl.appendChild(cell);
      }
    }

    kEpisode.textContent = String(episode);
    kEps.textContent = epsilon.toFixed(2);
  }

  async function animateRun(){
    agent = [...start];
    let totalReward = 0;
    let steps = 0;
    const maxSteps = 80;

    while (steps < maxSteps){
      const sKey = keyOf(agent);
      const qS = ensureQ(sKey);
      const a = argmax(qS); // purely exploit for the run
      const { next, reward } = step(a);

      agent = next;
      totalReward += reward;
      steps++;

      render();
      kSteps.textContent = String(steps);
      kReward.textContent = String(totalReward);

      if (isGoal(agent[0], agent[1])) break;
      await new Promise(res => setTimeout(res, 120));
    }
  }

  btnTrain100.addEventListener("click", () => {
    let last = { steps:0, totalReward:0 };
    for (let i=0;i<100;i++) last = trainEpisode();
    kSteps.textContent = String(last.steps);
    kReward.textContent = String(last.totalReward);
    render();
  });

  btnRun1.addEventListener("click", async () => {
    kSteps.textContent = "0";
    kReward.textContent = "0";
    render();
    await animateRun();
  });

  btnReset.addEventListener("click", () => {
    Q.clear();
    episode = 0;
    epsilon = 0.90;
    agent = [...start];
    kSteps.textContent = "0";
    kReward.textContent = "0";
    render();
  });

  // initial paint
  render();
})();