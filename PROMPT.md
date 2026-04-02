Создай симулятор five_link_robot с НУЛЯ:

АРХИТЕКТУРА (3 части):
├── sim_core/        # физика Rapier2D
├── api_server/      # HTTP + WebSocket  
└── web/             # HTML/CSS/JS (Canvas)

ШАГ 1 - ПАДАЮЩИЙ ШАР (проверка физики):
1. Шар: радиус 0.5м, масса 1кг, y=5м
2. Земля: y=0
3. Physics: 60 Hz, gravity -9.81
4. Консоль каждую секунду: "t=1s: y=3.52, vy=-2.94"

ШАГ 2 - ВЕБ-ИНТЕРФЕЙС:
web/
├── index.html       # страница
├── style.css        # стили
└── app.js           # Canvas 2D рендер

index.html:
- Canvas на весь экран
- Панель с кнопками: [Start] [Pause] [Reset]
- Отображение: t=2.34s, y=1.23m

app.js:
- fetch('/state') каждые 50ms
- Рисует шар: ctx.arc(x, y, radius)
- Камера: ctx.translate(0, canvas.height/2 - ball.y*100)

ШАГ 3 - API:
- GET  /state  -> {ball: {x, y, vy}, time}
- POST /reset  -> reset ball to y=5
- POST /pause  -> pause physics
- POST /resume -> resume physics
- WebSocket: ws://localhost:8080/ws (стриминг 30fps)

ШАГ 4 - РОБОТ (заменить шар):
- Тело: 0.3×0.15м, 2кг, y=1.0м
- 4 сустава: right_hip, right_knee, left_hip, left_knee
- Начальная поза: колени -1.2 рад
- Ползунки в вебе: -3.14 до +3.14
- API: POST /joint/angle {joint, angle}

ВЕБ ИНТЕРФЕЙС ДЛЯ РОБОТА:
┌─────────────────────────────────────┐
│  Canvas (робот в центре)            │
│  [====тело====]                     │
│     |        |                      │
│   [#нога#] [#нога#]                 │
│     |        |                      │
├─────────────────────────────────────┤
│ Right Hip: [====●====] 0.00 rad     │
│ Right Knee:[====●====] 0.00 rad     │
│ Left Hip:  [====●====] 0.00 rad     │
│ Left Knee: [====●====] 0.00 rad     │
│ [Reset] [Pause]                     │
└─────────────────────────────────────┘

ЗАВИСИМОСТИ:
[dependencies.sim_core]
rapier2d = "0.18"
nalgebra = "0.32"

[dependencies.api_server]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.5", features = ["cors", "fs"] }

КОД - минимальный:

// sim_core/src/lib.rs
pub struct Simulation {
    physics: PhysicsWorld,
    ball: Option<Ball>,
    robot: Option<Robot>,
    time: f64,
    paused: bool,
}

impl Simulation {
    pub fn new_ball() -> Self { ... }
    pub fn new_robot() -> Self { ... }
    pub fn step(&mut self, dt: f64) { ... }
    pub fn get_state(&self) -> Value { ... }
    pub fn set_joint(&mut self, name: &str, angle: f64) { ... }
}

// api_server/src/main.rs
#[tokio::main]
async fn main() {
    let sim = Arc::new(RwLock::new(Simulation::new_ball()));
    
    // Physics loop 60Hz
    tokio::spawn(run_physics(sim.clone()));
    
    // WebSocket broadcast
    tokio::spawn(broadcast_state(sim.clone()));
    
    let app = Router::new()
        .route("/state", get(get_state))
        .route("/reset", post(reset))
        .route("/joint/angle", post(set_joint))
        .nest_service("/", ServeDir::new("web"))
        .layer(CorsLayer::permissive());
    
    println!("Open: http://localhost:8080");
    axum::serve(listener, app).await;
}

// web/app.js
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function render() {
    const state = await fetch('/state').then(r => r.json());
    
    // Camera follows robot/ball
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.translate(canvas.width/2, canvas.height/2);
    ctx.scale(100, -100); // 100px = 1m, Y up
    ctx.translate(-state.ball.x, -state.ball.y);
    
    // Draw ground
    ctx.fillStyle = '#444';
    ctx.fillRect(-10, -0.1, 20, 0.1);
    
    // Draw ball
    ctx.beginPath();
    ctx.arc(state.ball.x, state.ball.y, 0.5, 0, Math.PI*2);
    ctx.fillStyle = '#ff6b6b';
    ctx.fill();
    
    requestAnimationFrame(render);
}

render();

ПРОВЕРКА:
1. cargo run -> http://localhost:8080
2. Видно падающий шар на Canvas
3. Кнопки Pause/Reset работают
4. Только потом заменить на робота

Начни с шага 1 - покажи падающий шар в браузере!
