namespace TCPServer
{
    public class Kotak
    {
        private readonly double[] lanes = { 100, 200, 300 }; // Left, Center, Right
        private int currentLaneIndex = 1; // start in center lane (lanes[1])

        public double X => lanes[currentLaneIndex];
        public double Y { get; private set; }
        public double Width { get; set; }
        public double Height { get; set; }

        private double groundY;
        private double jumpHeight = 30;
        private double slideHeight = 30;

        public Kotak(double y, double width = 50, double height = 50)
        {
            groundY = y;
            Y = y;
            Width = width;
            Height = height;
        }

        public void Move(string direction)
        {
            switch (direction.ToLower())
            {
                case "left":
                    if (currentLaneIndex > 0)
                        currentLaneIndex--;
                    break;

                case "right":
                    if (currentLaneIndex < lanes.Length - 1)
                        currentLaneIndex++;
                    break;

                case "up": // jump
                    Y = groundY - jumpHeight;
                    break;

                case "down": // slide
                    Y = groundY + slideHeight;
                    break;
            }

            // Kembalikan ke posisi semula setelah waktu singkat (simulate jump/slide)
            var timer = new System.Timers.Timer(300); // 300ms
            timer.Elapsed += (s, e) =>
            {
                Y = groundY;
                timer.Stop();
                timer.Dispose();
            };
            timer.Start();
        }
    }
}
