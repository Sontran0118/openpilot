// Peripheral init for the Nucleo-F446RE panda port.
// Link to Jetson = USART2 over the ST-Link VCP (PA2=TX, PA3=RX). No native USB used.
// CAN1 = PB8(RX)/PB9(TX) [AF9] -> transceiver 1 -> car main bus
// CAN2 = PB5(RX)/PB6(TX) [AF9] -> transceiver 2 -> car camera/LKAS bus

void gpio_usart2_init(void) {
  // PA2/PA3: USART2 (ST-Link VCP). AF7.
  set_gpio_alternate(GPIOA, 2, GPIO_AF7_USART2);
  set_gpio_alternate(GPIOA, 3, GPIO_AF7_USART2);
}

void gpio_can_init(void) {
  // CAN1: PB8 RX, PB9 TX (AF9)
  set_gpio_alternate(GPIOB, 8, GPIO_AF9_CAN1);
  set_gpio_alternate(GPIOB, 9, GPIO_AF9_CAN1);
  // CAN2: PB5 RX, PB6 TX (AF9)
  set_gpio_alternate(GPIOB, 5, GPIO_AF9_CAN2);
  set_gpio_alternate(GPIOB, 6, GPIO_AF9_CAN2);
}

// Common GPIO initialization
void common_init_gpio(void) {
  GPIOA->ODR = 0;
  GPIOB->ODR = 0;
  GPIOA->PUPDR = 0;
  GPIOB->AFR[0] = 0;
  GPIOB->AFR[1] = 0;

  gpio_usart2_init();
  gpio_can_init();

  // Nucleo LD2 user LED on PA5 (status)
  set_gpio_output(GPIOA, 5, false);
}

// USART2 baud config (APB1 = 45 MHz). panda uses 115200 for debug; the serial link runs faster.
// For the VCP link we use 1.5 Mbaud to keep CAN throughput. ST-Link VCP supports up to ~2 Mbaud.
#define SERIAL_BAUD 1500000U

void usart2_init(void) {
  // 8N1, TX+RX enabled, oversampling by 8 for high baud
  SERIAL_UART->CR1 = USART_CR1_OVER8;
  // BRR for OVER8: USARTDIV = fCK / (8 * baud); fCK = APB1 = 45 MHz
  // mantissa in [15:4], fraction (3 bits) in [2:0]
  {
    uint32_t fck = 45000000U;
    uint32_t div8 = (fck * 2U) / SERIAL_BAUD;   // = 2*fck/baud (for OVER8 scaling)
    uint32_t mant = div8 / 16U;
    uint32_t frac = div8 & 0x0FU;               // 4-bit -> use low 3 for OVER8
    SERIAL_UART->BRR = (mant << 4) | ((frac >> 1) & 0x07U);
  }
  SERIAL_UART->CR1 |= USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}

// Peripheral initialization
void peripherals_init(void) {
  // enable GPIO(A,B,C)
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOCEN;

  RCC->APB1ENR |= RCC_APB1ENR_PWREN;
  RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;

  // Connectivity: USART2 (VCP link) + CAN1 + CAN2
  RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
  RCC->APB1ENR |= RCC_APB1ENR_CAN1EN;
  RCC->APB1ENR |= RCC_APB1ENR_CAN2EN;   // CAN2 shares CAN1's clock domain (master/slave bxCAN)

  common_init_gpio();
  usart2_init();
}

// panda calls this to (re)bring the debug/comms uart. We route it to USART2.
void uart_init(USART_TypeDef *u, int baud) {
  UNUSED(u); UNUSED(baud);
  usart2_init();
}
